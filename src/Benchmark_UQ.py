import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats as sstats
import sys
import matplotlib as mpl

sys.path.insert(1, "/home/kevin.pasini/Workspace/n5_benchmark")
sys.path.insert(1, "/home/kevin.pasini/Workspace/n5_puncc")
sys.path.insert(1, "/home/kevin.pasini/Workspace/n5_uqmodels")

# Formalisation tache
from uqmodels.utils import compute_born, cut
from abench.benchmark import TimeSeries_from_dict,init_subpart
from abench.store import get_model_result,get_data_generator,get_cv_list,get_dataset

def calibrate_var(y, output, set_, mask, reduce, type_var="all", alpha=0.955, **kwarg):
    per_rejection = 1 - alpha
    pred, var_A, var_E = output
    if type_var == "epistemic":
        var = var_E
    elif type_var == "aleatoric":
        var = var_A
    elif type_var == "all":
        var = var_A + var_E

    Empirical_coverage = average_coverage(
        y, output, np.arange(len(y)), mask, reduce, type_var
    )
    y_lower, y_upper = compute_born(pred, np.sqrt(var), per_rejection)
    Empirical_coef = scipy.stats.norm.ppf(1 - ((1 - Empirical_coverage) / 2), 0, 1)
    True_coeff = scipy.stats.norm.ppf(1 - (per_rejection / 2), 0, 1)
    corr_ratio = np.power(True_coeff / Empirical_coef, 2)
    if type_var == "epistemic":
        new_output = output[0], output[1], output[2] * corr_ratio
    elif type_var == "aleatoric":
        new_output = output[0], output[1] * corr_ratio, output[2]
    elif type_var == "all":
        new_output = output[0], output[1] * corr_ratio, output[2] * corr_ratio
    return new_output


# Model wrapper :
class Encapsulated_model_UQ:
    def __init__(self, model):
        self.model = init_subpart(model)

    def _tuning(self, X, y, context=None, **kwarg):
        pass

    def fit(self, X, y, context=None, **kwarg):
        self.model.fit(X, y, context=None, **kwarg)

    def factory(self, X, y, mask=None, **kwarg):
        if hasattr(self.model, "factory"):
            Inputs, Outputs, mask = self.model.factory(X, y, mask, **kwarg)
        else:
            Inputs, Outputs = X, y
        return (Inputs, Outputs, mask)

    def predict(self, X, context=None, **kwarg):
        output = self.model.predict(X, context=None, **kwarg)
        if len(output[0].shape) == 1:
            output = [i.reshape(output[0].shape[0], 1) for i in output]
        return output

    def reset(self):
        if hasattr(self.model, "reset"):
            self.model.reset()

    def delete(self):
        if hasattr(self.model, "delete"):
            self.model.delete()
        else:
            del self.model


# Metrics tools & wrapper
def rmse(y, output, set_, mask, reduce, **kwarg):
    pred, var_A, var_E = output
    val = np.sqrt(np.power(pred[set_] - y[set_], 2).mean(axis=0))

    if mask:
        val = val[mask]

    if reduce:
        val = val.mean()
    return val


def dEIndicator(
    y, output, set_, mask, reduce, alpha=0.955, pen=0, type_var="all", **kwarg
):
    GIC_A = Gaussian_IC_errors(
        y, output, set_, mask, reduce, type_var="aleatoric", alpha=0.955, pen=0, **kwarg
    )
    GIC_E = Gaussian_IC_errors(
        y, output, set_, mask, reduce, type_var="epistemic", alpha=0.955, pen=0, **kwarg
    )
    return GIC_E / GIC_A


def Gaussian_NLL(y, output, set_, mask, reduce, type_var="all", mode=None, **kwarg):
    pred, var_A, var_E = output
    var_A, var_E = np.maximum(var_A, 0.00001), np.maximum(var_E, 0.00001)
    if type_var == "epistemic":
        sigma = np.sqrt(var_E)
        res = var_E
    elif type_var == "aleatoric":
        sigma = np.sqrt(var_A)
        res = (y - pred) ** 2
    elif type_var == "all":
        sigma = np.sqrt(var_A + var_E)
        res = (y - pred) ** 2

    val = -np.log(sigma) - 0.5 * np.log(2 * np.pi) - res / (2 * (sigma ** 2))
    val[val < -7] = -7
    if mask is None:
        val = val[set_].mean(axis=0)
    else:
        val = val[set_].mean(axis=0)[mask]
    if reduce:
        val = val.mean()
    return -val


def Gaussian_IC_errors(
    y, output, set_, mask, reduce, type_var="all", alpha=0.955, pen=0, **kwarg
):
    pred, var_A, var_E = output
    if type_var == "epistemic":
        var = var_E
    elif type_var == "aleatoric":
        var = var_A
    elif type_var == "all":
        var = var_A + var_E

    per_rejection = 1 - alpha
    y_lower, y_upper = compute_born(pred, np.sqrt(var), per_rejection)
    Empirical_sharpness = y_upper[set_] - y_lower[set_]

    Empirical_coverage = average_coverage(
        y, output, np.arange(len(y)), mask, reduce, type_var
    )
    new_var = calibrate_var(var, Empirical_coverage)
    y_lower, y_upper = compute_born(pred, np.sqrt(new_var), per_rejection)
    Corrected_sharpness = y_upper[set_] - y_lower[set_]

    Penalization = np.abs(Corrected_sharpness - Empirical_sharpness)
    val = Corrected_sharpness + pen * Penalization
    if mask is None:
        val = val.mean(axis=0)
    else:
        val = val.mean(axis=0)[mask]
    if reduce:
        val = val.mean()
    return val


def mae(y, output, set_, mask, reduce, **kwarg):
    pred, var_A, var_E = output

    if mask is None:
        val = np.abs(pred[set_] - y[set_]).mean(axis=0)
    else:
        val = np.abs(pred[set_] - y[set_]).mean(axis=0)[mask]

    if reduce:
        val = val.mean()
    return val


def cov_metrics(y, y_lower, y_upper, **kwarg):
    return ((y >= y_lower) & (y <= y_upper)).mean(axis=0)


def dEI(y, output, set_, mask, reduce, type_var="all", **kwarg):
    pred, var_A, var_E = output
    var_A, var_E = np.maximum(var_A, 0.00001), np.maximum(var_E, 0.00001)
    val = -0.5 * np.log(1 + (var_A[set_] / var_E[set_])).mean(axis=0)
    val = val
    if not (mask is None):
        val = val[mask]
    if reduce:
        val = val.mean()
    return val


def average_coverage(
    y, output, set_, mask, reduce, type_var="all", alpha=0.054, **kwarg
):
    pred, var_A, var_E = output
    if type_var == "epistemic":
        var = var_E
    elif type_var == "aleatoric":
        var = var_A
    elif type_var == "all":
        var = var_A + var_E
    y_lower, y_upper = compute_born(pred, np.sqrt(var), alpha)
    # var_A[var_A<0.01] = 0.01
    # var_E[var_E<0.001] = 0.001
    # anom_score = (np.abs(y - pred) + 0.5*np.sqrt(var_E)) / (2 * np.sqrt(var_A + var_E))
    if mask is None:
        val = cov_metrics(y[set_], y_lower[set_], y_upper[set_])
        # val = (anom_score[set_]<1).mean(axis=1)
    else:
        val = cov_metrics(y[set_], y_lower[set_], y_upper[set_])[mask]
        # val = (anom_score[set_]<1).mean(axis=1)[mask]
    if reduce:
        val = val.mean()
    return val


def sharpness(y, output, set_, mask, reduce, type_var="all", **kwarg):
    pred, var_A, var_E = output
    if type_var == "epistemic":
        var = var_E
    elif type_var == "aleatoric":
        var = var_A
    elif type_var == "all":
        var = var_A + var_E
    y_lower, y_upper = compute_born(pred, np.sqrt(var), 0.046)

    if mask is None:
        val = (y_upper[set_] - y_lower[set_]).mean(axis=0)
    else:
        val = (y_upper[set_] - y_lower[set_]).mean(axis=0)[mask]
    if reduce:
        val = val.mean()
    return val


def anom_score(
    y, output, set_, mask, reduce, type_var="all", min_A=0.08, min_E=0.02, **kwarg
):
    pred, var_A, var_E = output
    if type_var == "epistemic":
        var = var_E
    elif type_var == "aleatoric":
        var = var_A
    elif type_var == "all":
        var = var_A + var_E
        ind_A = np.sqrt(var_A)

    ind_E = np.sqrt(var_E)
    ind_E[ind_E < min_E] = min_E
    ind_A[ind_A < min_A] = min_A
    anom_score = (np.abs(y - pred) + ind_E) / (
        2 * np.sqrt(np.power(ind_E, 2) + np.power(ind_A, 2))
    )

    if mask is None:
        val = (anom_score[set_]).mean(axis=0)
    else:
        val = (anom_score[set_]).mean(axis=0)[mask]
    if reduce:
        val = val.mean()
    return val


def confidence_score(
    y, output, set_, mask, reduce, type_var="all", min_A=0.08, min_E=0.02, **kwarg
):
    pred, var_A, var_E = output
    if type_var == "epistemic":
        var = var_E
    elif type_var == "aleatoric":
        var = var_A
    elif type_var == "all":
        var = var_A + var_E

    ind_A = np.sqrt(var_A)
    ind_E = np.sqrt(var_E)
    ind_E[ind_E < min_E] = min_E
    ind_A[ind_A < min_A] = min_A
    confidence_score = ind_E / np.power(ind_A, 0.75)

    if mask is None:
        val = (confidence_score[set_]).mean(axis=0)
    else:
        val = (confidence_score[set_]).mean(axis=0)[mask]
    if reduce:
        val = val.mean()
    return val


# Visualisation tools


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

    k = 1
    if "k" in kwarg.keys():
        k = kwarg["k"]

    min_A, min_E = 0.1, 0.02
    if "var_min" in kwarg.keys():
        min_A, min_E = kwarg["var_min"]

    ind_A = np.sqrt(var_A)
    ind_E = np.sqrt(var_E)
    ind_E[ind_E < min_E] = min_E
    ind_A[ind_A < min_A] = min_A

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
        confidence_lvl += confidence_score > q

    plt.figure(figsize=size)
    plt.title(name)
    ax = plt.subplot(1, 1, 1)
    


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

    else:
        aux_plot_confiance(
            ax=ax,
            y_true=y[f_obs, dim],
            pred=pred[f_obs, dim],
            var_A=var_A[f_obs, dim],
            var_E=var_E[f_obs, dim],
            confidence_lvl=confidence_lvl[f_obs],
            mode_res=mode_res,
            **kwarg
        )

    if "ctx_attack" in kwarg.keys():
        y = y[f_obs, dim]
        dim_ctx, ctx_val = kwarg["ctx_attack"]
        if ctx_val == -1:
            list_ctx = list(set(context[f_obs, dim_ctx]))
            color = plt.get_cmap("jet", len(list_ctx))
            for n, i in enumerate(list_ctx):
                ax.fill_between(
                    f_obs - f_obs[0],
                    y.min(),
                    y.max(),
                    where=context[f_obs, dim_ctx] == i,
                    label=list_name_subset[int(i)],
                    color=color(n),
                    alpha=0.2,
                )

        else:
            ax.fill_between(
                f_obs - f_obs[0],
                y.min(),
                y.max(),
                where=context[f_obs, dim_ctx] == 1,
                color="yellow",
                alpha=0.2,
            )
    plt.legend(frameon=True, ncol=2, fontsize=14)
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
    ax.vlines(
        split_values,
        ylim_[0],
        ylim_[1],
        ls="--",
        color="black",
    )
    ax.set_ylim(ylim_[0], ylim_[1])
    ax.set_xlim(-0.5 + x.min(), x.max() + 0.5)
    ax.legend(ncol=7)


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

def compute_test(storing,list_exp,list_list_cv_attack,list_model,ind_ctx_attack):
    n_exp = len(list_exp)
    n_cv = len(list_list_cv_attack)
    res = np.zeros((len(list_model),n_cv,len(list_exp),2))
    cv_list = get_cv_list(storing)
    data_generator = get_data_generator(storing)
    for n_model,name_model in enumerate(list_model):
        list_x_plot  = []
        list_x_plot_bis  = []
        list_x_plot_ter  = []
        list_y_plot = []
        list_context  = []

        for cv,list_cv_attack in enumerate(list_list_cv_attack):
            name_ctx= ['low_var','mid_var','high_var']
            label = ['Sain','attack deg0 98','attack deg0 100']
            cl = ['green','orange','red']
            n_attack = len(list_cv_attack)
            list_output=[]
            for n_attack,cv_attack in enumerate(list_cv_attack):
                i = cv_list.index(cv_attack)
                X,y,split,context,_,cv_name = data_generator[i]
                test = (split==0)
                list_output.append(get_model_result(storing,name_model,cv_attack))

                var_A_controle = np.maximum(list_output[0][1], 0.0002)[:,0]
                var_E_controle = np.maximum(list_output[0][2], 0.0001)[:,0]
                var_A_controle,var_E_controle = cut(var_A_controle,0,0.995),cut(var_E_controle,0,0.995)
                ratio_controle = 0.5*np.log(1+(var_A_controle/var_E_controle))

                var_A_attacked =  np.maximum(list_output[-1][1], 0.0002)[:,0]
                var_E_attacked = np.maximum(list_output[-1][2], 0.0001)[:,0]
                var_A_attacked,var_E_attacked = cut(var_A_attacked,0,0.995),cut(var_E_attacked,0,0.995)
                ratio_attacked = 0.5*np.log(1+(var_A_attacked/var_E_attacked))
                delta_ratio = (np.copy(ratio_controle) - np.copy(ratio_attacked))

                mask_altered = (context[:,-1]==ind_ctx_attack[n_attack]) & test
                mask_normal = (context[:,-1]!=ind_ctx_attack[n_attack])  & test

                a = mask_altered.sum()
                b = mask_normal.sum()
                mu = (a*b)/2
                sigma = np.sqrt(((a*b)*(a+b+1))/12)
                test = sstats.mannwhitneyu(delta_ratio[mask_altered],delta_ratio[mask_normal],
                                           alternative='greater')
                res[n_model,cv,n_attack,0] = (test[0] - mu)/sigma

                test_controle = (sstats.mannwhitneyu(ratio_controle[mask_normal],ratio_controle[mask_altered],
                                                     alternative='greater')[0]-mu)/sigma
                test_attacked = (sstats.mannwhitneyu(ratio_attacked[mask_normal],ratio_attacked[mask_altered],
                                                     alternative='greater')[0]-mu)/sigma     
                res[n_model,cv,n_attack,1] = - (test_controle - test_attacked)
                if(cv==-1):
                    print(name_model,cv_attack,n_ctx_deg,
                          'TWC',res[n_model,cv,n_attack,0:2],
                          'Test_control',test_controle,test_attacked,test_attacked - test_controle,a)
    return(res)

def compute_test_inference(storing,list_exp,list_cv,list_model,n_injection=5):
    n_exp = len(list_exp)
    n_cv = len(list_cv)
    n_model = len(list_model)
    res = np.zeros((n_model,n_cv,n_exp+1,n_injection-1))
    
    def Wilocoxon_test(d1,d2):
        a = len(d1)
        b = len(d2)
        if(a!=b): #Echantillon different
            mu = (a*b)/2
            sigma = np.sqrt(((a*b)*(a+b+1))/12)
            stat_test = sstats.mannwhitneyu(d1,d2,alternative='greater')
            v = (stat_test[0]-mu)/sigma
        else: #Echantillon apparayé
            mu = (a*(a+1))/4
            sigma = np.sqrt((a*(a + 1)*(2*a + 1))/24)
            stat_test = sstats.wilcoxon(d1,d2,alternative='greater')
            v = (stat_test[0]-mu)/sigma
        return(v)

    for n_model,name_model in enumerate(list_model):
        list_x_plot  = []
        list_x_plot_bis  = []
        list_x_plot_ter  = []
        list_y_plot = []
        list_context  = []
        for cv,cv_name in enumerate(list_cv):
            X,y,split,context,_,cv_name = get_dataset(storing,cv_name)
            output = get_model_result(storing,name_model,cv_name)
            test = (split==0).reshape(n_injection,-1)[0]
            list_Var_A = np.maximum(output[1][:,0],0.0002).reshape(n_injection,-1)
            list_Var_E = np.maximum(output[2][:,0],0.0001).reshape(n_injection,-1)

            for n_attack,cv_attack in enumerate(list_exp):

                n_attack = n_attack+1
                var_A_controle = cut(list_Var_A[0],0,0.995)
                var_E_controle = cut(list_Var_E[0],0,0.995)
                ratio_controle = 0.5*np.log(1+(var_A_controle/var_E_controle))

                var_A_attacked = cut(list_Var_A[n_attack],0,0.995)
                var_E_attacked = cut(list_Var_E[n_attack],0,0.995)
                ratio_attacked = 0.5*np.log(1+(var_A_attacked/var_E_attacked))
                res[n_model,cv,n_attack,0] = Wilocoxon_test(ratio_controle[test],ratio_attacked[test])  
                for n_ctx,ctx in enumerate(set(context[:,-1])):
                    flag = ((context[:,-1].reshape(5,-1)[0])==ctx) & (test)
                    res[n_model,cv,n_attack,1+n_ctx] =Wilocoxon_test(ratio_controle[flag],
                                                                     ratio_attacked[flag])
    return(res)
