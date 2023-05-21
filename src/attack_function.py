import numpy as np

# Attack à l'apprentissage

# Randomly drop
def attack_classic(n,drop_rate):
    return(np.random.uniform(0,1,n)<drop_rate)

# Auxilliar seq constraint drop function
def attack_on_seq(n,drop_rate,len_past=0,len_futur=1,stick_seq=1,seed=0):
    keep_sample = np.zeros(n).astype(bool)
    Extended_sample = np.zeros(n).astype(bool)
    cpt = 0
    np.random.seed(seed+cpt)
    list_idx = np.random.choice(np.arange(n)[len_past:-(len_futur+stick_seq)], 
                                size=int((n*(1-drop_rate)/(len_past+len_futur+stick_seq))), replace=False)
    
    for idx in list_idx:
        keep_sample[idx:(idx+stick_seq)]=1
        Extended_sample[idx-len_past:(idx+len_futur+stick_seq)]=1
    
    mask_to_choose = np.invert(keep_sample)[len_past:-(len_futur+stick_seq)]
    list_of_choices = np.arange(len_past,n-(len_futur+stick_seq))[mask_to_choose].tolist()  
    while (keep_sample.mean() <= (1-drop_rate)) & (list_of_choices != []):
        cpt += 1
        np.random.seed(seed+cpt)
        idx = np.random.choice(list_of_choices,size=1, replace=False)
        for i in range(-len_past,1+stick_seq+len_futur):
            if(idx[0]+i in list_of_choices):
                list_of_choices.remove(idx[0]+i) 
        #keep_sample[idx[0]:(idx[0]+1+stick_seq)]=1
        keep_sample[idx[0]-len_past:(idx[0]+len_futur+stick_seq)]=1
    if((keep_sample.mean() <= (1-drop_rate))):
        print('Warning, Selected paramater lead to %keep max of  :',keep_sample.mean())
    if(drop_rate==1):
        keep_sample=keep_sample*0
    return(np.invert(keep_sample))

        
# Seq attack inference generic
def ctx_seq_attack(y,context,ctx_deg,drop_rate=0.7,add_ctx=-1,mode='seq',
                   len_past=12,len_futur=3,stick_seq=0):
    remove_from_train = np.zeros(len(y))
    for ctx_attacked in list(set(context[context[:,-1]==ctx_deg,add_ctx])):
        flag_ctx = context[:,add_ctx] == ctx_attacked
        if(mode=='seq'):
            remove_from_train[flag_ctx] = attack_on_seq(n=flag_ctx.sum(),
                                                        drop_rate=drop_rate,
                                                        len_past=len_past,
                                                        len_futur=len_futur,
                                                        stick_seq=stick_seq).astype(bool)
        else: #random
            remove_from_train[flag_ctx] = attack_classic(flag_ctx.sum(),drop_rate=drop_rate)

    return(remove_from_train)

# Seq attack inference for UC air liquide
def ctx_seq_attack_UC_AL(y,context,ctx_deg,drop_rate = 0.7,add_ctx=-1,mode='seq',
                         len_past=12,len_futur=3,stick_seq=0):
    
    remove_from_train = np.zeros(len(y))
    list_ctx_attacked = list(set(context[context[:,-1]==ctx_deg,1]))
    for ctx_attacked in list_ctx_attacked:
        flag_ctx = context[:,1] == ctx_attacked
        if(mode=='seq'):
            remove_from_train[flag_ctx] = attack_on_seq(n=flag_ctx.sum(),
                                                        drop_rate=drop_rate,
                                                        len_past=len_past,
                                                        len_futur=len_futur,
                                                        stick_seq=stick_seq).astype(bool)
        else: #random
            remove_from_train[flag_ctx] = attack_classic(flag_ctx.sum(),drop_rate=drop_rate)
    return(remove_from_train)

# Test and print ctx train, test and drop %
def test_attack(dataset_generator,n_ctx=-1):
    print('% train for each sub sample for each set')
    for n,(_,_,split_,context_,_,cv_name) in enumerate(dataset_generator):
        train_val = []
        test_val = []
        drop_val = []
        for i in list(set(context_[:,n_ctx])):
            flag = context_[:,n_ctx] == i
            train_val.append((split_[flag]==1).sum()/len(split_)*100)
            test_val.append((split_[flag]==0).sum()/len(split_)*100)
            drop_val.append((split_[flag]==-1).sum()/len(split_)*100)
        print(cv_name,' len',len(split_),
              '%train/test/drop:',np.round(train_val,1),np.round(test_val,1),np.round(drop_val,1))

#############################################################
# Attack à l'inférence
################################################

#Attack for binary feature
def swap_bin(X,n_attack,mode='small'):
    X = np.copy(X)
    if(n_attack > X.shape[1]):
        n_attack = X.shape[1]
    for ind in range(X.shape[0]):
        f_ind=np.random.choice(X.shape[1],n_attack,replace=False)
        for f in f_ind:
            if(mode=='small'):
                X[ind,f] = 1 - X[ind,f]
            else:
                X[ind] = np.zeros(X.shape[1])
    return(X)
 
#Attack for numeric feature
def swap_num(X,context_altered,attack_values,n_attack,mode='small',f_attack=None):
    cpt=0
    X = np.copy(X)
    cpt_chg=np.zeros(len(X))
    if(n_attack > X.shape[1]):
        n_attack = X.shape[1]
    for ctx in list(set(context_altered)):
        mask=(context_altered==ctx)
        for ind in np.nonzero(mask)[0]:
            if(f_attack):
                f_ind = np.random.choice(f_attack,n_attack,replace=False)
            else:
                f_ind= np.random.choice(X.shape[1],n_attack,replace=False)
            for f in f_ind:
                cpt+=1
                if(mode=='small'):
                    X[ind,f]= np.maximum(attack_values[int(ctx),f],X[ind,f])
                else:
                    X[ind,f]= np.maximum(attack_values[-1,f],X[ind,f])
    return(X)

#Pre calculated remplacement valiues
def compute_stat_recap(X,context_altered,mode='small',force=1):
    n_ctx = len(set(context_altered))
    if(mode=='small'):
        attack_values = np.zeros((n_ctx,X.shape[1],2))
        for n,i in enumerate(list(set(context_altered))):
            mask = context_altered==i
            for j in range(X.shape[1]):
                attack_values[n,j,0]=np.quantile(X[mask,j],0.10,axis=0)
                attack_values[n,j,1]=np.quantile(X[mask,j],0.90,axis=0)
    else:
        attack_values = np.zeros((1,X.shape[1],2))
    for j in range(X.shape[1]):
        attack_values[-1,j,0]=np.quantile(X[:,j],0.10,axis=0)
        attack_values[-1,j,1]=np.quantile(X[:,j],0.90,axis=0)
        
    attack_values_tot = attack_values[:,:,1] + force * (attack_values[:,:,1]-attack_values[:,:,0])
    return(attack_values_tot)

#Attack function decicated to AC features
def attack_AL(X,context,n_attack=1,mode='small'):
    X_o_1,X_o_2,X_o_3,X_t,X_n=X[:,:9],X[:,9:13],X[:,13:15],X[:,15:18],X[:,18:]
    attack_values = compute_stat_recap(X_n,context)
    X_o_1 = swap_bin(X_o_1,n_attack,mode)
    X_o_2 = swap_bin(X_o_2,n_attack,mode)
    X_o_3 = swap_bin(X_o_3,n_attack,mode)
    X_n = swap_num(X_n,context,attack_values,n_attack,mode)
    return(np.concatenate([X_o_1,X_o_2,X_o_3,X_t,X_n],1))

#Attack function decicated to AC features
def attack_bis(X,context,list_f_to_attack,list_f_nature,n_attack=1,mode='small',force=1):
    X=np.copy(X)
    print(mode)
    if(n_attack>0):
        attack_values = compute_stat_recap(X,context,mode=mode,force=force)
        X = swap_num(X,context,attack_values,n_attack=n_attack,mode=mode,f_attack=list_f_to_attack)
    return(X)

def injection_on_data_generator(data_generator,type_attack,type_f,list_f,force):
        new_data_generator = []
        for X,y,split,context,obj_param,cv_name in data_generator:
            X_new = X
            y_new = y
            split_new = split
            context_new = np.concatenate([context[:,:-2],
                                          np.zeros(len(X))[:,None],
                                          context[:,-1][:,None]],axis=1)
            test = (split==0)
            for n,(mode,n_swap) in enumerate(type_attack):
                context_altered = (context[:,0]%80)//10
                X_new_attack = attack_bis(X,context_altered,list_f,type_f,n_swap,mode,force=force)
                X_new = np.concatenate([X_new,X_new_attack],axis=0)
                print(X_new[:,list_f].mean())
                y_new = np.concatenate([y_new,y],axis=0)
                split_new = np.concatenate([split_new,-split],axis=0)
                context[:,-2] = n+1
                context_new = np.concatenate([context_new,context],axis=0)
            new_data_generator.append([X_new,y_new,split_new,context_new,obj_param,cv_name])
        return(new_data_generator)