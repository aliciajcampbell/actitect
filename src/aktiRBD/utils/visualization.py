from dataclasses import dataclass
from datetime import datetime, timedelta, time
from typing import Union

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics

from aktiRBD import utils

__all__ = ['draw_actigraphy_data', 'draw_roc_or_pr_curve', 'draw_cv_roc_or_pr_curve', 'draw_cv_boxplot']

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'


def draw_actigraphy_data(df: pd.DataFrame, _sleep_log: pd.DataFrame, raw_only: bool = False):
    @dataclass
    class Colors:
        x: str = '#6200EE'
        y: str = '#03DAC6'
        z: str = '#bb86fc'
        sptw: str = 'gray'
        sb: str = 'gray'
        nw: str = 'r'

    max_value, min_value = df[['x', 'y', 'z']].max().max(), df[['x', 'y', 'z']].min().min()
    maxrange = max(max_value, -1 * min_value)
    minrange = -1 * maxrange

    grouped_days = df.groupby(df.index.date)
    nrows = len(grouped_days) + 1

    fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex=False, sharey=True, figsize=(10, nrows), dpi=100)
    plt.subplots_adjust(wspace=.1)

    lbl_ax = fig.add_subplot(111, frameon=False)
    lbl_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    lbl_ax.grid(False)
    lbl_ax.set_xlabel(r'time / h', fontsize=12, labelpad=15)
    lbl_ax.set_ylabel(r'acceleration magnitude / $g$', fontsize=12, labelpad=20)

    i = 1
    for day, group in grouped_days:
        ax = axes[i]

        # ax.plot(group.index.to_numpy(), group['acc'].to_numpy(), c='k', zorder=200, alpha=.5)
        ax.plot(group.index.to_numpy(), group['x'].to_numpy(), c=Colors.x, zorder=2000, lw=1.4, alpha=1)
        ax.plot(group.index.to_numpy(), group['y'].to_numpy(), c=Colors.y, zorder=2000, lw=1, alpha=.95)
        ax.plot(group.index.to_numpy(), group['z'].to_numpy(), c=Colors.z, zorder=2000, lw=.6, alpha=.85)

        if not raw_only:
            ax.fill_between(
                group.index.to_numpy(), maxrange, minrange, where=group['sptw'], facecolor=Colors.sptw, alpha=.3)
            ax.fill_between(
                group.index.to_numpy(), maxrange, minrange, where=group['sleep_bout'], facecolor=Colors.sb, alpha=.5)
            ax.fill_between(group.index.to_numpy(), maxrange, minrange,
                            where=~(group['wear'].to_numpy()), facecolor=Colors.nw, alpha=.3)

        if isinstance(_sleep_log, pd.DataFrame):
            if not _sleep_log.empty:
                _log_entries = []
                for _val in _sleep_log.values.squeeze()[3:]:
                    if isinstance(_val, datetime):
                        if _val.date() == day:
                            _log_entries.append(_val)

                if _log_entries:
                    for _le in _log_entries:
                        if isinstance(_le, pd.Timestamp):
                            _le = _le.to_pydatetime()
                        ax.axvline(_le, color='k', zorder=2000, lw=5)
                        ax.axvline(_le, color='w', zorder=2100, lw=1, alpha=.8)

        ax.yaxis.set_label_position('right')
        ax.set_ylabel(day.strftime("%A\n%d %B"), weight='bold', ha='left', va='center', rotation=0, fontsize='medium',
                      color='k', labelpad=10)
        ax.get_xaxis().grid(True, which='major', color='k', alpha=.75, lw=.75, zorder=100)
        ax.get_xaxis().grid(True, which='minor', color='grey', alpha=.25, lw=.75, zorder=100)

        ax.set_xlim(day, day + timedelta(days=1))
        ax.set_ylim(minrange, maxrange)
        ax.set_xticks(pd.date_range(
            start=datetime.combine(day, time(0, 0, 0, 0)),
            end=datetime.combine(day + timedelta(days=1), time(0, 0, 0, 0)),
            freq='4h'))
        ax.set_xticks(pd.date_range(
            start=datetime.combine(day, time(0, 0, 0, 0)),
            end=datetime.combine(day + timedelta(days=1), time(0, 0, 0, 0)),
            freq='1h'), minor=True)
        ax.tick_params(left=True, right=False, top=False, bottom=True)
        ax.tick_params(which='major', direction='out', length=5, width=2)
        # ax.minorticks_on()
        ax.tick_params(which='minor', direction='out', length=3, width=1)

        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_color('k')
        ax.spines['left'].set_zorder(1000)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color('k')
        ax.spines['bottom'].set_zorder(1000)
        ax.spines['bottom'].set_linewidth(1.2)

        ax.spines['right'].set_color('darkgrey')
        ax.spines['top'].set_color('darkgrey')

        ax.set_facecolor('whitesmoke')
        i += 1

    leg_ax = axes[0]
    leg_ax.axis('off')
    legend_patches = [mlines.Line2D([], [], visible=False, label='Legend:'),
                      mlines.Line2D([], [], color=Colors.x, lw=2, label=r'$\vec{a}_{x}$'),
                      mlines.Line2D([], [], color=Colors.y, lw=2, label=r'$\vec{a}_{y}$'),
                      mlines.Line2D([], [], color=Colors.z, lw=2, label=r'$\vec{a}_{z}$'),
                      mlines.Line2D([], [], color='k', lw=5, label=r'sleep log entry')]
    if not raw_only:
        legend_patches.append(mlines.Line2D([], [], color=Colors.sptw, lw=5, alpha=.3, label=r'sleep window'))
        legend_patches.append(mlines.Line2D([], [], color=Colors.sb, lw=5, alpha=.5, label=r'sleep bout'))
        legend_patches.append(mlines.Line2D([], [], color=Colors.nw, lw=5, label=r'non-wear'))

    leg_ax.legend(handles=legend_patches, bbox_to_anchor=(0., 0., 1., 1.), loc='center', ncol=8, mode="best",
                  borderaxespad=0, framealpha=0.6, frameon=True, fancybox=True)

    fig.autofmt_xdate()
    axes[-1].set_xticklabels(['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'],
                             fontweight='bold', fontsize='medium', ha='center')
    axes[1].set_xticklabels(['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'],
                            fontweight='bold', fontsize='small', ha='center')
    axes[1].xaxis.set_tick_params(pad=-4)
    axes[-1].tick_params(rotation=0)
    axes[1].tick_params(rotation=0, bottom=True, top=False, labeltop=True)

    # fig.tight_layout()
    del grouped_days
    del df
    return fig


def draw_roc_or_pr_curve(x: np.ndarray, y: np.ndarray, thres: np.ndarray, mode: dict,
                         opt_thres: Union[float, dict] = None):
    assert 'curve' in mode and mode['curve'] in ['roc', 'pr'], "mode must contain 'curve' with value 'roc' or 'pr'"
    assert 'lvl' in mode and mode['lvl'] in ['night', 'patient'], \
        "mode must contain 'lvl' with value 'night' or 'patient'"
    if mode['curve'] == 'pr':
        assert 'pos_frac' in mode and isinstance(mode['pos_frac'], float), \
            "mode must contain 'pos_frac' as a float when 'curve' is 'pr'"

    _c = 'b' if mode['curve'] == 'roc' else 'deeppink'
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
    plt.subplots_adjust(left=.17, right=.97, top=.95, bottom=.14)
    ax.set_title(f"{mode['lvl']} classification", fontweight='bold', pad=7, fontsize=13)

    ax.scatter(x, y, s=20, marker='o', zorder=200, color=_c, alpha=.6, edgecolor='k')

    ax.set_xlim(-.045, 1.045)
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=2, top=True, right=True,
                   labelsize=15)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1, top=True, right=True)

    if mode['curve'] == 'roc':
        if opt_thres:
            _opt_idx = np.where(opt_thres == thres)
            ax.scatter(x[_opt_idx], y[_opt_idx], s=70, marker='*', zorder=220, color='limegreen', alpha=1.0,
                       edgecolor='k', label=f"op. point={opt_thres:.2f}")

        ax.plot(x, y, lw=1, zorder=200, color=_c, linestyle='-', alpha=1.0,
                label=f"{r'$AUC=$'}{metrics.auc(x, y):.2f}")
        ax.plot([0, 1.], [0, 1], color='k', lw=1, linestyle='--')
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=16, labelpad=20)
        ax.set_xlabel('False Positive Rate (1-Specifity)', fontsize=16, labelpad=20)
        ax.legend(loc=(.4, .1), fontsize=15, frameon=False)

    elif mode['curve'] == 'pr':

        f1_scores = np.divide(
            2 * (x * y), x + y,
            out=np.zeros_like(x),  # set default to zero, i.e. when pr. = rec. = 0
            where=(x + y) != 0  # avoid division by zero
        )

        _pos_frac = mode['pos_frac']

        if opt_thres:
            for _name, _opt_thresh in opt_thres.items():
                _opt_idx = np.where(_opt_thresh == thres)
                ax.scatter(x[_opt_idx], y[_opt_idx], s=70, marker='*', zorder=220, color='limegreen', alpha=1.0,
                           edgecolor='k', label=f"op. point ={_opt_thresh:.2f} ({_name})")

        ax.plot(x, y, lw=1, zorder=200, color=_c, linestyle='-', alpha=1.0,
                label=f"{r'$f_{1}^{ max}=$'}{np.nanmax(f1_scores):.2f}")
        ax.plot([0, 1.], [_pos_frac, _pos_frac], color='k', lw=1, linestyle='--', label=f'pos. frac={_pos_frac:.2f}')
        ax.set_ylabel('Precision', fontsize=16, labelpad=20)
        ax.set_xlabel('Recall', fontsize=16, labelpad=20)
        ax.set_ylim(_pos_frac - .045, 1.045)
        ax.legend(loc=(.4, .75), fontsize=15, frameon=False)

    return fig


def draw_cv_roc_or_pr_curve(curve_dict: dict[list], mode: str, cl: float = .90, display_op_point: bool = False):
    if mode == 'roc':
        x, y = np.array(curve_dict['fpr']), np.array(curve_dict['tpr'])
        metric, op_point = np.array(curve_dict['auc']), np.array(curve_dict['op_point'])
    elif mode == 'pr':
        x, y = np.array(curve_dict['rec']), np.array(curve_dict['prec'])
        metric = np.array(curve_dict['f1_max'])
        op_point = {'dist': np.array(curve_dict['op_point_dist']), 'f1': np.array(curve_dict['op_point_f1'])}
    else:
        raise ValueError(f"mode bust be 'roc' or 'pr', got {mode}.")

    mean_y, std_y = np.mean(y, axis=0), np.std(y, axis=0)
    _cl_std = stats.norm.ppf((1 + cl) / 2)
    y_upper = np.minimum(mean_y + _cl_std * std_y, 1)
    y_lower = np.maximum(mean_y - _cl_std * std_y, 0)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
    fig.set_facecolor('none')
    plt.subplots_adjust(left=.17, right=.97, top=.95, bottom=.14)
    ax.set_xlim(-.045, 1.045)
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=2, top=True, right=True, labelsize=15)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1, top=True, right=True)

    mean, std, ci_lower, ci_upper, _ = utils.compute_mean_std_ci(metric, confidence_level=cl)

    _c = 'b' if mode == 'roc' else 'deeppink'
    # _label_line = f"{r'$AUC=$'}{np.mean(metric):.2f}±{np.std(metric):.2f}" if mode == 'roc' \
    #    else f"{r'$f1_{max}=$'}{np.mean(metric):.2f}±{np.std(metric):.2f}"

    _metric_name = 'f1_{max}' if mode == 'pr' else 'AUC'
    _line_label = rf"${_metric_name} = {mean:.2f}^{{\,+{ci_upper - mean:.2f}}}_{{\,-{mean - ci_lower:.2f}}}$"

    ax.plot(x, mean_y, lw=1, zorder=200, color=_c, linestyle='-', alpha=1.0, label=_line_label)
    ax.fill_between(x, y_lower, y_upper, zorder=150, color='grey', alpha=.2,
                    label=rf"$ {cl * 100:.0f}\% \,CI\,(\sigma={std:.2f})$")
    if display_op_point:
        _label_scatter = f"op. point={np.mean(op_point):.2f}±{np.std(op_point):.2f}" if mode == 'roc' \
            else (f"op. point={np.mean(op_point['dist']):.2f}±{np.std(op_point['dist']):.2f} "
                  f"({np.mean(op_point['f1']):.2f}±{np.std(op_point['f1']):.2f} f1)")
        ax.scatter(x, mean_y, s=20, marker='o', zorder=200, color=_c, alpha=.6, edgecolor='k', label=_label_scatter)

    for _y in y:
        ax.plot(x, _y, lw=.7, zorder=100, color='grey', linestyle='-', alpha=.35)

    if mode == 'roc':
        ax.set_ylim(-.045, 1.045)
        mean_y[-1] = 1.0
        ax.plot([0, 1.], [0, 1], color='k', lw=1, linestyle='--')
        ax.legend(loc=(.4, .1), fontsize=15, frameon=False)
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=16, labelpad=20)
        ax.set_xlabel('False Positive Rate (1-Specifity)', fontsize=16, labelpad=20)

    elif mode == 'pr':
        _mean_pos_frac = np.min(mean_y)
        ax.set_ylim(.5, 1.02)
        ax.legend(loc=(.05, .05), fontsize=15, frameon=False)
        ax.set_ylabel('Precision', fontsize=16, labelpad=20)
        ax.set_xlabel('Recall', fontsize=16, labelpad=20)
        ax.plot([0, 1], [_mean_pos_frac, _mean_pos_frac], color='k', lw=1, linestyle='--',
                label=f"{_mean_pos_frac:.2f} pos. frac.")

    return fig


def draw_cv_boxplot(history: dict, ylims=(.5, 1), cl: float = .90):
    names, data = [], []
    for name, _data in history.items():
        if name == 'balanced_accuracy':
            name = 'bal. accuracy'
        elif name == 'average_precision':
            name = 'avg. precision'
        elif name == 'roc_auc':
            name = 'auroc'

        if name not in ('class_thresh', 'mean', 'std'):
            names.append(name)
            data.append(_data.flatten())

    def _pick_colors_from_colormap(colormap_name, k):
        cmap = plt.get_cmap(colormap_name)
        return np.array([cmap(j / (k - 1)) for j in range(k)])

    colors = _pick_colors_from_colormap('Greens', len(names))
    medians = np.median(data, axis=1)
    sorted_indices = np.argsort(medians)
    sorted_colors = np.empty(len(names), dtype=object)
    for i, index in enumerate(sorted_indices):
        sorted_colors[index] = colors[i]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.set_facecolor('none')
    plt.subplots_adjust(left=.04, right=.99, top=.98, bottom=.06)

    bplot = ax.boxplot(data, labels=names, notch=True, patch_artist=True, sym='k+', zorder=200)
    for patch, color in zip(bplot['boxes'], sorted_colors):
        patch.set_facecolor(color)
        patch.set_alpha(.82)
    median_color = 'k'
    for median in bplot['medians']:
        median.set_color(median_color)
        median.set_linewidth(1.5)

    _y_offset = .05
    for i, (_name, _data) in enumerate(zip(names, data)):
        lower_whisker = bplot['whiskers'][2 * i].get_ydata()[1]
        mean, std, ci_lower, ci_upper, _ = utils.compute_mean_std_ci(_data, confidence_level=cl)
        _label = '\n'.join(names[i].split()) + '\n' \
                 + r'$' + f'{mean:.2f}^{{+{ci_upper - mean:.2f}}}_{{-{mean - ci_lower:.2f}}}' + r'$' \
                 + '\n' + r'$\sigma = ' + f'{std:.2f}' + r'$'
        ax.text(i + 1, lower_whisker - _y_offset, _label, ha='center', va='top', fontsize=15, color='k')

    ax.tick_params(axis='both', which='major', direction='in', length=6, width=2, top=True, right=True,
                   labelsize=12)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1, top=True, right=True)

    ax.grid(axis='y', which='major', c='grey', alpha=.4, lw=.5)
    # ax.set_ylim(max(0, np.min(data)-.2), 1)
    ax.set_ylim(*ylims)
    ax.set_xlim(.25, len(names) + .75)
    ax.set_xticklabels([])
    # ax.set_yticks(np.linspace(.5, 1.0, 6))
    return fig
