import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
from uqmodels.utils import cut,compute_born
from abench.store import get_dataset,get_model_result

def Synthesis_dUQ_test(list_res,list_model,pos,subtitle,list_exp,colors):
    n_exp = len(list_exp)
    fig,ax = plt.subplots(2,2,figsize=(15,5), gridspec_kw={'width_ratios': [2.5, 1]})

    plt.style.use('seaborn-darkgrid')
    color_test=['darkred','limegreen']
    plt.suptitle('Sample shift due to injection (corrected by original sample shift)'
                 +'on real data for learning injection',fontsize=11)
    for n,res_c in enumerate(list_res):
        n_model,n_cv,n_exp,n_stat = res_c.shape
        ax0 = plt.subplot(2,2,n+1)
        plt.title(subtitle[n],fontsize=17)
        for m in range(len(list_model)):
            for i in np.arange(n_exp-1)+1:
                for k in range(4):
                    x = i+pos[m]
                    plt.scatter(x,res_c[m,k,i,1],s=7,color=color_test[res_c[m,k,i,1]>2],zorder=100)
                plt.errorbar(y=res_c[m,:,i,1].mean(axis=0),x=x,yerr=res_c[m,:,i,1].std(axis=0),
                             capsize=5,color=colors[m],marker="d")
            plt.errorbar(y=[],x=[],yerr=[],capsize=5,color=colors[m],marker="d",label=list_model[m])
            plt.xticks(np.arange(len(list_exp[n]))+1,list_exp[n],rotation=0)
        #plt.yscale("symlog")
        plt.scatter([],[],s=7,color='limegreen',zorder=100,label='Cv_succes')
        plt.scatter([],[],s=7,color='darkred',zorder=100,label='Cv_faillure')
        plt.hlines(2,1-0.3,n_exp-0.7,ls=':',color='red',label='2σ')
        plt.hlines(4,1-0.3,n_exp-0.7,ls=':',color='darkred',label='4σ')
        plt.xlim(0.7,n_exp-1+0.3)
        #plt.legend(fontsize=10, bbox_to_anchor=(0.9,-0.23),ncol=8,frameon=True)

        plt.subplot(2,2,2+n+1)
        #plt.title('Sample shift due to injection (corrected by original sample shift) on real data for learning injection',fontsize=11)
        for m in range(len(list_model)):
            for i in np.arange(n_exp-1)+1:
                for k in range(4):
                    x = i+pos[m]
                    plt.scatter(x,res_c[m,k,i,0],s=7,color=color_test[res_c[m,k,i,0]>2],zorder=100)
                plt.errorbar(y=res_c[m,:,i,0].mean(axis=0),x=x,yerr=res_c[m,:,i,0].std(axis=0),
                             capsize=5,color=colors[m],marker="d")
            plt.errorbar(y=[],x=[],yerr=[],capsize=5,color=colors[m],marker="d",label=list_model[m])
        plt.scatter([],[],s=10,color='limegreen',zorder=100,label='Cv_succes')
        plt.scatter([],[],s=7,color='darkred',zorder=100,label='Cv_faillure')
        plt.hlines(2,1-0.3,n_exp-0.7,ls=':',color='red',label='2σ')
        plt.hlines(4,1-0.3,n_exp-0.7,ls=':',color='darkred',label='4σ')
        plt.xticks(np.arange(len(list_exp[n]))+1,list_exp[n],rotation=0)   

        #plt.yscale("symlog")
        #plt.suptitle('Model deviation due to injection against control model deviation on real data for learning injection',fontsize=11)
        plt.xlim(0.7,n_exp-1+0.3)
    plt.legend(fontsize=10, bbox_to_anchor=(0.9,-0.23),ncol=8,frameon=True)
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.8,wspace=0.1,hspace=0.2)
    plt.text(.5, 0.01, 'Model deviation due to injection against control model deviation'
             +'on real data for learning injection', fontsize=11, transform=fig.transFigure, horizontalalignment='center')
    plt.show()

def HP_studies_plot(res_b,res,list_name,colors,list_marker,pos):
    plt.subplots(2,2,figsize=(12,5), gridspec_kw={'height_ratios':[0.7, 2]})
    plt.subplot(2,2,1)
    plt.style.use('seaborn-darkgrid')
    n_model,n_variant = res_b.shape[0],res_b.shape[1]
    n_cv,n_exp = res.shape[1],res.shape[2]-1
    plt.title('RMSE on train-set',fontsize=17)
    for m in range(n_model):
        for i in range(n_variant):
            x=i+pos[m]
            plt.errorbar(y=res_b[m,i,0,0],x=x,yerr=res_b[m,i,0,1],capsize=5,
                         color=colors[m],marker=list_marker[i-1])
        plt.errorbar(y=[],x=[],yerr=[],capsize=5,color=colors[m],label=list_name[m*3+1])
    #plt.plot(np.array(pos[0:5])+1,res_b[0:5,i,0].mean(axis=1),color=colors[0],ls='--',zorder=-10)
    #plt.plot(np.array(pos[5:10])+1,res_b[5:10,i,0].mean(axis=1),color=colors[1],ls='--',zorder=-10)
    #plt.plot(np.array(pos[10:15])+1,res_b[10:15,i,0].mean(axis=1),color=colors[2],ls='--',zorder=-10)
    plt.xticks([0,1,2,3,4],['very low','low','normal','high','very high'],fontsize=12)
    plt.xlabel('Ratio Complexity/Variability',fontsize=13, y=-0.1)
    plt.ylim(0.09,0.26)
    plt.subplot(2,2,2)
    plt.title('RMSE on test-set',fontsize=17)
    pos=[-0.2,0,0.2]
    for m in range(n_model):
        for i in range(n_variant):
            x=i+pos[m]
            plt.errorbar(y=res_b[m,i,1,0],x=x,yerr=res_b[m,i,1,1],capsize=5,
                         color=colors[m],marker=list_marker[i-1])
        plt.errorbar(y=[],x=[],yerr=[],capsize=5,color=colors[m],marker=list_marker[i-1])
    plt.tight_layout()
    plt.xticks([0,1,2,3,4],['very low','low','normal','high','very high'],fontsize=12)
    plt.xlabel('Ratio Complexity/Variability',fontsize=13, y=-0.1)
    plt.ylim(0.09,0.26)
    plt.grid(True)


    color_test=['darkred','limegreen']
    mu = 0
    sigma = 1
    seuil = scipy.stats.norm.ppf(0.95,loc=mu,scale=sigma)
    seuil2 = scipy.stats.norm.ppf(0.997,loc=mu,scale=sigma)
    list_marker = ["d","d","d","d","d","d"]
    pos= [-0.1,1-0.1,2-0.1,3-0.10,4-0.1,0,1,2,3,4,0.1,1.1,2.1,3.1,4.1]
    colors = ['red','blue','green']
    for m in range(n_model*n_variant):
        plt.subplot(2,2,3)
        plt.title('Test T1B : Subset shift due to injection',fontsize=17, y=-0.15)
        for i in np.arange(n_exp)+1:
            for k in range(n_cv):
                x=i+pos[m]
                #plt.scatter(x,res[m,k,i,0],s=10,color=color_test[res[m,k,i,0]>seuil],zorder=100)
            plt.errorbar(y=res[m,:,i,1].mean(axis=0),x=x,yerr=res[m,:,i,1].std(axis=0),
                         capsize=5,color=colors[m//5],marker=list_marker[i-1])
        plt.xticks(np.array([0,1,2,3,4])+1,['very low','low','normal','high','very high'],y=1.1,fontsize=12)
        if(m==0):
            plt.hlines(seuil,0.8,5.2,ls=':',color='red',label='2σ boundary')
            plt.hlines(seuil2,0.8,5.2,ls=':',color='darkred',label='3σ boundary')
    plt.legend(fontsize=10)
    plt.plot(np.array(pos[0:5])+1,res[0:5,:,i,1].mean(axis=1),color=colors[0],ls='--',zorder=-10)
    plt.plot(np.array(pos[5:10])+1,res[5:10,:,i,1].mean(axis=1),color=colors[1],ls='--',zorder=-10)
    plt.plot(np.array(pos[10:15])+1,res[10:15,:,i,1].mean(axis=1),color=colors[2],ls='--',zorder=-10)
    plt.ylim()
    plt.grid(True)

    for m in range(n_model*n_variant):
        plt.subplot(2,2,4)
        plt.title('Test T2 : model deviation due to injection',fontsize=17, y=-0.15)
        for i in np.arange(n_exp)+1:
            for k in range(n_cv):
                x=i+pos[m]
                #plt.scatter(x,res[m,k,i,1],s=10,color=color_test[res[m,k,i,1]>seuil],zorder=100)
            if(m%5==2):
                    plt.errorbar(y=[],x=[],capsize=5,color=colors[m//5],
                                 marker=list_marker[i-1],label=list_name[m])
            plt.errorbar(y=res[m,:,i,0].mean(axis=0),x=x,yerr=res[m,:,i,0].std(axis=0),
                         capsize=5,color=colors[m//5],marker=list_marker[i-1])

        if(m==0):
            plt.hlines(seuil,0.8,5.2,ls=':',color='red')
            plt.hlines(seuil2,0.8,5.2,ls=':',color='darkred')
            plt.xticks(np.array([0,1,2,3,4])+1,['very low','low','normal','high','very high'],
                       y=1.1,fontsize=12)
        plt.tight_layout()
        plt.grid(True)
    plt.plot(np.array(pos[0:5])+1,res[0:5,:,i,0].mean(axis=1),color=colors[0],ls='--',zorder=-10)
    plt.plot(np.array(pos[5:10])+1,res[5:10,:,i,0].mean(axis=1),color=colors[1],ls='--',zorder=-10)
    plt.plot(np.array(pos[10:15])+1,res[10:15,:,i,0].mean(axis=1),color=colors[2],ls='--',zorder=-10)
    plt.legend()
    plt.show()
    
def plot_confidence(
    y,
    output,
    output_2=None,
    list_set=None,
    fig_name="test",
    type_var_E=1,
    ylim=None,
    figsize=(15, 4),
    split_values=-1,
    x=None,
):
    list_result = [output]
    n_model = 1
    if not (output_2 is None):
        n_model = 2
        list_result = [output, output_2]

    fig, axes = plt.subplots(nrows=len(list_set) * n_model, ncols=1, figsize=figsize)
    if list_set is None:
        list_set = [np.ones(len(y)) == 1]

    for i, set_ in enumerate(list_set):
        y_true = y[set_, 0]
        for j, result in enumerate(list_result):
            if len(list_set) * n_model == 1:
                ax = axes
            else:
                ax = axes[i * n_model + (j)]
            pred, var_A, var_E = (
                result[0][:, 0][set_],
                cut(result[1][:, 0], 0.01, 0.98)[set_],
                cut(result[2][:, 0], 0.01, 0.98)[set_],
            )
            if type_var_E == 1:
                color = np.log(var_E) - 0.5 * np.log(var_A)
                color = np.power(color + color.min(), 2)

            aux_plot_confiance(ax, y_true, pred, var_A, color, ylim, split_values, x=x)

            if i * n_model + (j) == 0:
                ax.legend(ncol=4, loc="upper left")
            if i * n_model + (j) != len(list_set) * n_model - 1:
                ax.set_xticklabels([])

        # ax.set_title('Operational context '+str(i+1),fontsize=13)
    cbar_ax = fig.add_axes([0.25, 0.123, 0.50, 0.015])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=plt.get_cmap("RdYlGn_r")),
        orientation="horizontal",
        fraction=0.2,
        pad=0.1,
        cax=cbar_ax,
    )
    cbar.set_label("Epistemic_confidence", labelpad=-10, y=0.45, fontsize=13)

    fig.subplots_adjust(bottom=0.2, hspace=0.04, wspace=0.05)
    cbar.set_ticks([0.1, 0.5, 0.9])
    cbar.set_ticklabels(["high", "", "low"])
    # plt.tight_layout()
    fig.savefig(fig_name, bbox_inches="tight")


def uncertainty_plot(
    y, output, train, test, context, size, f_obs, name, mode_res, dim=0, **kwarg
):
    pred, var_A, var_E = output
    if type(f_obs) == type(None):
        f_obs = np.arange(len(y))[test]

    ind_ctx=None
    if "ind_ctx" in kwarg.keys():
        ind_ctx = kwarg["ind_ctx"]
        
    split_ctx = -1
    if "split_ctx" in kwarg.keys():
        split_ctx = kwarg["split_ctx"]
        
    ylim = None
    if "ylim" in kwarg.keys():
        ylim = kwarg["ylim"]
        
    if "compare_deg" in kwarg.keys():
        compare_deg =  kwarg["compare_deg"]
    else:
        compare_deg = False

    min_A, min_E = 0.1, 0.01
    if "var_min" in kwarg.keys():
        min_A, min_E = kwarg["var_min"]

    var_E[var_E < min_E] = min_E
    var_A[var_A < min_A] = min_A

    confidence_score = np.exp(-0.5 * np.log(1 + var_A[:, 0] / var_E[:, 0]))
    confidence_lvl = np.zeros(len(confidence_score))
    local_conf = False
    if "local_conf" in kwarg.keys():
        conf_score = kwarg["local_conf"]
    
    only_data = False
    if "only_data" in kwarg.keys():
        only_data = kwarg["only_data"]
        list_name_subset = kwarg["list_name_subset"]
        if(only_data):
            name='Data'

    for n, i in enumerate([0.50, 0.80, 0.90, 0.95, 0.975, 0.99]):
        q = np.quantile(confidence_score[f_obs], i)
        if compare_deg :
            q = np.quantile(confidence_score[f_obs][:f_obs[int(len(f_obs)/2)]], i)
        confidence_lvl += confidence_score > q
    
    plt.figure(figsize=size)
    plt.title(name)
    f_obs_full=np.copy(f_obs)
    n_ctx = 1
    if (split_ctx>-1):
        if(ind_ctx is None):
            list_ctx_ = list(set(context[f_obs,split_ctx]))
        else:
            list_ctx_ = ind_ctx
        n_ctx = len(list_ctx_)
        
    for n_fig in range(n_ctx):
        ax = plt.subplot(1, n_ctx, n_fig+1)
        if (split_ctx>-1):
            f_obs = f_obs_full[context[f_obs_full,split_ctx]==list_ctx_[n_fig]]
        if only_data:
            ax.scatter(
                f_obs - f_obs[0],
                y[f_obs, dim],
                c="black",
                s=10,
                marker="x",
                linewidth=1,
                label="observation",
            )

            ax.plot(
                f_obs - f_obs[0],
                y[f_obs, dim],
                ls="-",
                color="darkgreen",
                alpha=1,
                linewidth=0.7,
                zorder=-4,
            )
            if not(ylim is None):
                ax.set_ylim(ylim[0],ylim[1])
            else:
                y_lim=(y.min(),y.max())

        else:
            aux_plot_confiance(
                ax=ax,
                y_true=y[f_obs, dim],
                pred=pred[f_obs, dim],
                var_A=var_A[f_obs, dim],
                var_E=var_E[f_obs, dim],
                confidence_lvl=confidence_lvl[f_obs],
                mode_res=mode_res,
                **kwarg)

        if "ctx_attack" in kwarg.keys():
            y_ = y[f_obs, dim]
            if (ylim is None):
                ylim=(y.min(),y.max())
            dim_ctx, ctx_val = kwarg["ctx_attack"]
            if ctx_val == -1:
                list_ctx = list(set(context[f_obs, dim_ctx]))
                color = plt.get_cmap("jet",len(list_name_subset))
                list_color = [color(i) for i in range(3)]
                for n, i in enumerate(list_ctx):
                    ax.fill_between(
                        f_obs - f_obs[0],
                        ylim[0],
                        ylim[1],
                        where=context[f_obs, dim_ctx] == i,
                        color=list_color[int(i)],
                        alpha=0.2)
            else:
                ax.fill_between(
                    f_obs - f_obs[0],
                    y.min(),
                    y.max(),
                    where=context[f_obs, dim_ctx] == 1,
                    color="yellow",
                    alpha=0.2)
                
        if(n_fig!=0):
            ax.set_yticklabels([])
            
        if "ctx_attack" in kwarg.keys():
            color = plt.get_cmap("jet",len(list_name_subset))
            for n, i in enumerate(range(len(list_name_subset))):
                ax.fill_between([],ylim[0],ylim[1],
                    where=[],
                    label=list_name_subset[int(i)],
                    color=color(i),
                    alpha=0.2)
    plt.suptitle(name)
    plt.subplots_adjust(wspace=0.03,hspace=0.03,left=0.1, bottom=0.22, right=0.90, top=0.8)
    plt.legend(frameon=True, ncol=6, fontsize=14,bbox_to_anchor=(0.5, 0, 0.38,-0.11))
    # plt.xlim(0, 8400)

    cmap = [plt.get_cmap("RdYlGn_r", 7)(i) for i in [0, 1, 2, 3, 4, 5, 6]]
    bounds = [0, 4, 6, 7, 8, 9, 9.5, 10]
    cmap = mpl.colors.ListedColormap(cmap)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    color_ls = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar1 = plt.colorbar(
        color_ls,
        pad=0.05,
        shrink=1.3,
        orientation="vertical",
        spacing="proportional",
    )
    cbar1.set_label("Confidence_lvl", fontsize=14)
    cbar1.set_ticks([2, 5.5, 6.5, 7.5, 8.5, 9.25, 9.75])
    cbar1.set_ticklabels(["50%", "30%", "10%", "5%", "2.5%", "1.5%", "1%"], fontsize=12)
    plt.tight_layout()
    plt.show()
    return


def aux_plot_confiance(
    ax,
    y_true,
    pred,
    var_A,
    var_E,
    confidence_lvl,
    ylim=None,
    split_values=-1,
    x=None,
    mode_res=False,
    min_A=0.08,
    min_E=0.02,
    **kwarg
):

    if x is None:
        x = np.arange(len(y_true))

    if mode_res:
        y_true = y_true - pred
        pred = pred * 0

    y_lower_A, y_up_A = compute_born(pred, np.sqrt(var_A), 0.045)
    y_lower_E, y_upper_E = compute_born(pred, np.sqrt(var_E), 0.045)
    y_lower, y_upper = compute_born(pred, np.sqrt(var_A + var_E), 0.045)
    y_lower_N, y_upper_N = compute_born(pred, np.sqrt(var_A + var_E), 0.32)

    ax.plot(x, y_true, ls="dotted", color="black", linewidth=0.9, alpha=1)
    ax.plot(
        x,
        y_true,
        ls="dotted",
        color="black",
        linewidth=0.9,
        alpha=1,
    )
    ax.plot(
        x,
        pred,
        "-",
        color="darkgreen",
        alpha=1,
        linewidth=0.7,
        zorder=-4,
        label="Prediction",
    )

    ax.fill_between(
        x,
        y_upper_N,
        y_lower_N,
        color="darkblue",
        alpha=0.2,
        label="Interval 1σ (68%)",
    )
    ax.fill_between(
        x,
        y_lower,
        y_upper,
        color="teal",
        alpha=0.2,
        label="Interval 2σ (95%)",
    )
    if False:
        ax.plot(x, y_lower_A, color="blue", ls="dotted", lw=1.2, label="Var_A")
        ax.plot(x, y_lower_E, color="green", ls="dotted", lw=1.2, label="Var_E")
        ax.plot(x, y_upper_A, color="blue", ls="dotted", lw=1.2)
        ax.plot(x, y_upper_E, color="green", ls="dotted", lw=1.2)

    ind_A = np.sqrt(var_A)
    ind_E = np.sqrt(var_E)
    ind_E[ind_E < min_E] = min_E
    ind_A[ind_A < min_A] = min_A
    anom_score = (np.abs(y_true - pred) + 0*ind_E) / (2 * np.sqrt(np.power(ind_E, 2) + np.power(ind_A, 2)))
    flag_anom = anom_score > 1

    label = "Prediction"
    for i in range(0, 1 + int(confidence_lvl.max())):
        mask = i == confidence_lvl
        ax.scatter(
            x[mask],
            pred[mask],
            c=confidence_lvl[mask],
            marker="D",
            s=14,
            edgecolors="black",
            linewidth=0.2,
            cmap=plt.get_cmap("RdYlGn_r"),
            vmin=0,
            vmax=int(confidence_lvl.max()),
            label=label,
            zorder=10 + i,
        )
        label = None

    ax.scatter(
        x,
        y_true,
        c="black",
        s=10,
        marker="x",
        linewidth=1,
        label="real demand",
    )
    ax.scatter(
        x[flag_anom],
        y_true[flag_anom],
        linewidth=1,
        marker="x",
        c="magenta",
        s=25,
        label='"Abnormal" real demand',
    )

    if ylim is None:
        ylim_ = min(y_true.min(), y_lower.min()), max(y_true.max(), y_upper.max())
        ylim_ = ylim_[0] + np.abs(ylim_[0] * 0.05), ylim_[1] - np.abs(ylim_[1] * 0.05)
    else:
        ylim_ = ylim[0], ylim[1]

    ax.set_ylim(ylim_[0], ylim_[1])
    ax.set_xlim(-0.5 + x.min(), x.max() + 0.5)
    #ax.legend(ncol=7)


from matplotlib import gridspec
from scipy import stats


def plot_density(
    x_plot, y_plot, context, label, cl, name_ctx=[""], suptitle="", pb_cut=5
):
    categories = len(label)
    list_ctx = list(set(context))
    n_ctx = len(list_ctx)
    # Set up 4 subplots as axis objects using GridSpec:
    gs = gridspec.GridSpec(
        2, 2 * n_ctx, width_ratios=[1, 3] * n_ctx, height_ratios=[3, 1]
    )
    # Add space between scatter plot and KDE plots to accommodate axis labels:
    gs.update(hspace=0.3, wspace=0.3)

    # Set background canvas colour to White instead of grey default
    fig = plt.figure(figsize=(5 * n_ctx, 5))
    plt.suptitle(suptitle, fontsize=15)
    fig.patch.set_facecolor("white")

    for n, ctx in enumerate(list_ctx):
        mask_ctx = context == ctx
        x_plot_ctx = x_plot[:, mask_ctx]
        y_plot_ctx = y_plot[:, mask_ctx]

        plt.title(name_ctx[n], fontsize=10)
        ax = plt.subplot(
            gs[0, 1 + n * 2]
        )  # Instantiate scatter plot area and axis range

        ax.set_xlim(x_plot_ctx.min(), x_plot_ctx.max())
        ax.set_ylim(y_plot_ctx.min(), y_plot_ctx.max())
        ax.set_xlabel("logratio E/A", zorder=100)
        ax.set_ylabel("P(Y|context)", zorder=100)

        axl = plt.subplot(gs[0, 0 + n * 2], sharey=ax)  # Instantiate left KDE plot area
        axl.get_xaxis().set_visible(False)  # Hide tick marks and spines
        axl.get_yaxis().set_visible(False)
        axl.spines["right"].set_visible(False)
        axl.spines["top"].set_visible(False)
        axl.spines["bottom"].set_visible(False)

        axb = plt.subplot(
            gs[1, 1 + n * 2], sharex=ax
        )  # Instantiate bottom KDE plot area
        axb.get_xaxis().set_visible(False)  # Hide tick marks and spines
        axb.get_yaxis().set_visible(False)
        axb.spines["right"].set_visible(False)
        axb.spines["top"].set_visible(False)
        axb.spines["left"].set_visible(False)

        axc = plt.subplot(gs[1, 0 + n * 2])  # Instantiate legend plot area
        axc.axis("off")  # Hide tick marks and spines

        # Plot data for each categorical variable as scatter and marginal KDE plots:

        for n in range(categories):
            ax.scatter(
                x_plot_ctx[n, :],
                y_plot_ctx[n, :],
                color="none",
                label=label[n],
                s=0.5,
                edgecolor=cl[n],
                alpha=0.5,
                zorder=-n,
            )

            kde = stats.gaussian_kde(x_plot_ctx[n, :])
            xx = np.linspace(x_plot_ctx.min(), x_plot_ctx.max(), 500)

            axb.plot(xx, kde(xx), color=cl[n])
            axb.set_ylim(0, min(pb_cut, kde(xx).max()))
            kde = stats.gaussian_kde(y_plot_ctx[n, :])
            yy = np.linspace(y_plot_ctx.min(), y_plot_ctx.max(), 500)
            axl.errorbar(kde(yy), yy, None, color=cl[n], elinewidth=0.1)
            axl.set_xlim(0, min(pb_cut, kde(yy).max()))

    # Copy legend object from scatter plot to lower left subplot and display:
    # NB 'scatterpoints = 1' customises legend box to show only 1 handle (icon) per label
    handles, labels = ax.get_legend_handles_labels()
    axc.legend(handles, labels, scatterpoints=1, loc="center", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_density_cv(
    list_x_plot,
    list_y_plot,
    list_context,
    label,
    cl,
    name_ctx=[""],
    suptitle="",
    pb_cut=5,
    xlabel="x",
    ylabel="y",
    only_ctx=False,
    same_x_scale=False,
    same_y_scale=False,
):
    categories = len(label)
    list_ctx = list(set(list_context[0]))
    n_ctx = len(list_ctx)
    # Set up 4 subplots as axis objects using GridSpec:

    x_min_ctx = np.min([x_plot.min() for x_plot in list_x_plot])
    x_max_ctx = np.min([x_plot.max() for x_plot in list_x_plot])
    y_min_ctx = np.min([y_plot.min() for y_plot in list_y_plot])
    y_max_ctx = np.max([y_plot.max() for y_plot in list_y_plot])
    if only_ctx:
        fig = plt.figure(figsize=(7, 7))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 3], height_ratios=[3, 1])
    # Add space between scatter plot and KDE plots to accommodate axis labels:
    else:
        gs = gridspec.GridSpec(
            2, 2 * n_ctx, width_ratios=[1, 3] * n_ctx, height_ratios=[3, 1]
        )
        fig = plt.figure(figsize=(7 * n_ctx, 4))
    gs.update(hspace=0.3, wspace=0.3)
    # Set background canvas colour to White instead of grey default
    plt.suptitle(suptitle, fontsize=15)
    fig.patch.set_facecolor("white")

    if only_ctx:
        ax = plt.subplot(gs[0, 1])
        axl = plt.subplot(gs[0, 0], sharey=ax)
        axb = plt.subplot(gs[1, 1], sharex=ax)
        axc = plt.subplot(gs[1, 0])

    for n, ctx in enumerate(list_ctx):
        # Instantiate scatter plot area and axis range
        if not (only_ctx):
            ax = plt.subplot(gs[0, 1 + n * 2])
            axl = plt.subplot(gs[0, 0 + n * 2], sharey=ax)
            axb = plt.subplot(gs[1, 1 + n * 2], sharex=ax)
            axc = plt.subplot(gs[1, 0 + n * 2])
            list_x_min = []
            list_x_max = []
            list_y_min = []
            list_y_max = []
        else:
            color = cl[n]

        ax.set_title(name_ctx[n], fontsize=10)
        ax.set_xlabel(xlabel, zorder=100)
        ax.set_ylabel(ylabel, zorder=100)
        axl.get_xaxis().set_visible(False)  # Hide tick marks and spines
        axl.get_yaxis().set_visible(False)
        axl.spines["right"].set_visible(False)
        axl.spines["top"].set_visible(False)
        axl.spines["bottom"].set_visible(False)

        axb.get_xaxis().set_visible(False)  # Hide tick marks and spines
        axb.get_yaxis().set_visible(False)
        axb.spines["right"].set_visible(False)
        axb.spines["top"].set_visible(False)
        axb.spines["left"].set_visible(False)
        axc.axis("off")  # Hide tick marks and spines

        # Plot data for each categorical variable as scatter and marginal KDE plots:
        list_kde_xx = []
        list_kde_yy = []
        cptt = 0
        for c in range(categories):
            if not (only_ctx):
                color = cl[c]
            label_ = None
            if cptt == 0:
                cptt += 1
                label_ = label[c]
            list_kde_xx.append([])
            list_kde_yy.append([])
            for x_plot, y_plot, context in zip(list_x_plot, list_y_plot, list_context):
                mask_ctx = context == ctx
                x_plot_ctx = cut(x_plot[:, mask_ctx], 0.01, 0.99)
                y_plot_ctx = cut(y_plot[:, mask_ctx], 0.01, 0.99)

                if not (only_ctx):
                    list_x_min.append(x_plot_ctx.min())
                    list_x_max.append(x_plot_ctx.max())
                    list_y_min.append(y_plot_ctx.min())
                    list_y_max.append(y_plot_ctx.max())

                ax.scatter(
                    x_plot_ctx[c, :],
                    y_plot_ctx[c, :],
                    color="none",
                    s=0.3,
                    edgecolor=color,
                    label=label_,
                    alpha=0.4,
                    zorder=-c,
                )

                list_kde_xx[c].append(stats.gaussian_kde(x_plot_ctx[c, :]))
                list_kde_yy[c].append(stats.gaussian_kde(y_plot_ctx[c, :]))

        if not (same_x_scale):
            x_min_ctx, x_max_ctx = np.min(list_x_min), np.max(list_x_max)

        if not (same_y_scale):
            y_min_ctx, y_max_ctx = np.min(list_y_min), np.max(list_y_max)

        xx = np.linspace(x_min_ctx, x_max_ctx, min(100, len(x_plot_ctx[c])))
        yy = np.linspace(y_min_ctx, y_max_ctx, min(100, len(y_plot_ctx[c])))

        pb_xx_max = []
        pb_yy_max = []
        for c in range(categories):
            cpt = 0
            label_ = None
            if cpt == 0:
                cpt += 1
                label_ = label[c]

            if not (only_ctx):
                color = cl[c]
            list_density_xx = []
            list_density_yy = []
            for kde_xx, kde_yy in zip(list_kde_xx[c], list_kde_yy[c]):
                tab_val = kde_xx(xx)
                list_density_xx.append(tab_val / tab_val.sum())
                tab_val = kde_yy(yy)
                list_density_yy.append(tab_val / tab_val.sum())
            list_density_xx = np.array(list_density_xx)
            list_density_yy = np.array(list_density_yy)

            if False:
                axb.errorbar(
                    xx,
                    list_density_xx.mean(axis=0),
                    yerr=list_density_xx.std(axis=0),
                    color=color,
                    elinewidth=0.5,
                    capsize=2,
                    alpha=0.6,
                )
            else:
                axb.plot(
                    xx, list_density_xx.mean(axis=0), ls="--", color=color, label=label_
                )
                axb.fill_between(
                    xx,
                    list_density_xx.mean(axis=0) - list_density_xx.std(axis=0),
                    list_density_xx.mean(axis=0) + list_density_xx.std(axis=0),
                    color=color,
                    ls=":",
                    alpha=0.2,
                )

            pb_xx_max.append(list_density_xx.mean(axis=0).max())

            if False:
                axl.errorbar(
                    list_density_yy.mean(axis=0),
                    yy,
                    xerr=list_density_yy.std(axis=0),
                    color=color,
                    elinewidth=0.5,
                    capsize=2,
                    alpha=0.6,
                )
            else:
                axl.plot(list_density_yy.mean(axis=0), yy, ls="--", color=color)
                axl.fill_betweenx(
                    yy,
                    list_density_yy.mean(axis=0) - list_density_yy.std(axis=0),
                    list_density_yy.mean(axis=0) + list_density_yy.std(axis=0),
                    color=color,
                    ls=":",
                    alpha=0.2,
                )

            pb_yy_max.append(list_density_yy.mean(axis=0).max())

        axl.set_xlim(0, min(pb_cut, np.max(pb_yy_max)))
        ax.set_xlim(x_min_ctx, x_max_ctx)
        ax.set_ylim(y_min_ctx, y_max_ctx)
        axb.set_ylim(0, min(pb_cut, np.max(pb_xx_max)))
        handles, labels = ax.get_legend_handles_labels()
        axc.legend(handles, labels, scatterpoints=1, loc="center", fontsize=10)

    # Copy legend object from scatter plot to lower left subplot and display:
    # NB 'scatterpoints = 1' customises legend box to show only 1 handle (icon) per label
    plt.tight_layout()
    plt.show()

    
def dUQ_space_plot(storing,list_cv_comparison,list_subset_name,list_name_norm):
    def log_scale(a):
        return(np.log(a-a.min(axis=0)+1))

    def Q_colors(array):
        colors = np.zeros(len(array)) 
        for i in range(0,100):
            colors[array>np.quantile(array,(i/100))] = i
        return(colors)

    ctx_name = list_subset_name
    for name_model in list_name_norm:
        fig, ax = plt.subplots(1,5,subplot_kw=dict(projection='3d'),figsize=(20,6))
        plt.suptitle(name_model+' on real data with learning injection on mid-var subset')
        for n,cv_attack in enumerate(list_cv_comparison):
            X,y,split,context,_,cv_name = get_dataset(storing,cv_attack)
            test = (split==0)
            pred,var_A,var_E = np.array(get_model_result(storing,name_model,cv_attack,))[:,:,0]     
            var_A, var_E = np.maximum(var_A, 0.0002), np.maximum(var_E, 0.0001)
            var_A = cut(var_A,0,0.995)
            var_E = cut(var_E,0,0.995)

            flag = test #| np.invert(test)
            indicator = - np.log(1+(var_A/var_E))[flag]
            ctx = context[flag,-1]
            x, y, z = pred[flag],var_A[flag],np.exp(cut((indicator),0,0.999))
            vmin4 = 25
            vmax4 = 100 
            if(n==0):
                indicator_control = np.copy(indicator)

            if(n==1):
                indicator_control = Q_colors((indicator - indicator_control))
                im=ax[4].scatter(x, y, z, c=indicator_control,s=4,alpha=0.8,
                                 vmin=vmin4,vmax=vmax4,cmap='turbo')
                cbar = plt.colorbar(im,ax=ax[4],cmap='Spectral',label="dEI diff quantile",fraction=0.1,
                                    pad=0.1,shrink=0.50, anchor=(0.5,0.5),orientation='horizontal')
                cbar.set_ticks([25,50,75,100])
                cbar.set_ticklabels(['25<%','50%','75%','100%'])

            ax[0].set_title('Subsets location for control model')
            ax[1].set_title('dE-indicator for Control model')
            ax[2].set_title('Subsets location for Degraged model')
            ax[3].set_title('dE-indicator for Degraged model')
            ax[4].set_title('dE-indicator diffrence between models')

            im=ax[(n*2)].scatter(x, y, z, c=ctx,s=4,alpha=0.3,cmap=plt.get_cmap('jet',len(ctx_name)))
            cbar3 = plt.colorbar(im,ax=ax[(n*2)],cmap='jet',label="data",
                                 fraction=0.1,pad=0.1,shrink=0.50, anchor=(0.5, 0.5),orientation='horizontal')
            cbar3.set_ticks(np.arange(len(ctx_name)))
            cbar3.set_ticklabels(ctx_name)

            im=ax[(n*2)+1].scatter(x, y, z,
                                   c=Q_colors(indicator),s=2,alpha=0.8,vmin=vmin4,vmax=vmax4,cmap='turbo')
            cbar = fig.colorbar(im,ax=ax[(n*2)+1],label="dEI Ranking",
                                fraction=0.1,pad=0.1,shrink=0.50, anchor=(0.5, 0.5),orientation='horizontal')
            cbar.set_ticks([25,50,75,100])
            cbar.set_ticklabels(['25<%','50%','75%','100%'])
            elev, azim = 15,-75
            if(n==0):
                set_=[0,1]
            else:
                set_=[2,3,4]
            for j in set_:
                ax[j].view_init(elev, azim)
                ax[j].set_xlim3d(x.min(),x.max())
                ax[j].set_ylim3d(y.min(),y.max())
                ax[j].set_zlim3d(z.min(),z.max())
                ax[j].set_xlabel('Prediction') #'Log-(residu²/σ)')
                ax[j].set_ylabel('σ-Aleatoric')
                ax[j].set_zlabel('exp(dEI)')
        plt.tight_layout()

def Synthesis_dUQ_test_inference(list_res,list_model,pos,subtitle,list_exp,colors):
    n_res = len(list_res)
    plt.subplots(1,2,figsize=(15,3), gridspec_kw={'width_ratios': [2, 2]})
    plt.suptitle('Test of Sample shift due to inference injection',fontsize=20)
    color_test=['darkred','limegreen']
    for n,res_c in enumerate(list_res):
        ax0 = plt.subplot(1,n_res,n+1)
        plt.style.use('seaborn-darkgrid')
        n_exp = len(list_exp)
        n_cv=res_c.shape[1]
        colors = ['red','blue','green','orange','cyan','lime']
        plt.style.use('seaborn-darkgrid')
        color_test=['darkred','limegreen']
        plt.title(subtitle[n],fontsize=17)
        n_model,n_cv,n_exp,n_stat = res_c.shape
        for m in range(len(list_model)):
            for i in np.arange(n_exp-1)+1:
                ind=0
                for k in range(n_cv):
                    x = i+pos[m]
                    plt.scatter(x,res_c[m,k,i,ind],s=7,color=color_test[res_c[m,k,i,ind]>2],zorder=100)
                plt.errorbar(y=res_c[m,:,i,ind].mean(axis=0),x=x,yerr=res_c[m,:,i,ind].std(axis=0),
                             capsize=5,color=colors[m],marker="d")
            plt.errorbar(y=[],x=[],yerr=[],capsize=5,color=colors[m],marker="d",label=list_model[m])
            plt.xticks(np.arange(len(list_exp[n]))+1,list_exp[n],rotation=0)
        plt.scatter([],[],s=7,color='limegreen',zorder=100,label='Cv_succes')
        plt.scatter([],[],s=7,color='darkred',zorder=100,label='Cv_faillure')
        plt.hlines(2,1-0.3,n_exp-0.7,ls=':',color='red',label='2σ')
        plt.hlines(4,1-0.3,n_exp-0.7,ls=':',color='darkred',label='4σ')
        plt.subplots_adjust(wspace=0.06,left=0.1, bottom=0.22, right=0.90, top=0.8)
        plt.xlim(0.7,n_exp-1+0.3)
    plt.legend(fontsize=11,ncol=8,frameon=True,bbox_to_anchor=(0.5, 0, 0.38,-0.11))
    plt.show()