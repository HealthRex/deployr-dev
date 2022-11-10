"""
Defines a series a suite of classes that evaluate performance of silently
deployed models. Base class is BinaryEvaluator.  
"""
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import (precision_score,
                             recall_score,
                             accuracy_score,
                             average_precision_score,
                             precision_recall_curve,
                             roc_auc_score,
                             roc_curve
                             )

sns.set_theme(style='whitegrid', font_scale=2.0)


THRESHOLD_DEPENDENT = ['accuracy_score', 'recall_score', 'precision_score']


class BinaryEvaluator:

    def __init__(self, outdir):
        self.outdir = outdir
        self.metrics = {
            'Accuracy': accuracy_score,
            'Sensitivity': recall_score,
            'Specificity': recall_score,
            'Precision': precision_score,
            'AUROC': roc_auc_score,
            'Average precision': average_precision_score
        }
        os.makedirs(outdir, exist_ok=True)

    def __call__(self, labels, predictions):
        """
        Override in child classes to include functionality to pull labels and
        predictions from some outside source (ex: cosmos db)
        """
        self.get_performance_artifacts(labels, predictions)

    def get_performance_artifacts(self, labels, predictions):
        """
        Computes a suite of performance measures and saves artifacts
        """
        results = self.bootstrap_metric(labels, predictions, self.metrics)
        with open(os.path.join(self.outdir, "metrics.json"), "w") as fp:
            json.dump(results, fp)

        fig, axs = plt.subplots(1, 3, figsize=(30, 10))
        self.plot_roc_curve(labels=labels,
                            predictions=predictions,
                            title='ROC Curve',
                            ax=axs[0])
        self.plot_precision_recall(labels=labels,
                                   predictions=predictions,
                                   title='PR Curve',
                                   ax=axs[1])
        self.plot_calibration_curve(labels=labels,
                                    predictions=predictions,
                                    title='Calibration Curve',
                                    ax=axs[2])
        plt.savefig(os.path.join(self.outdir, 'performance_curves.png'),
                    bbox_inches='tight',
                    dpi=300)

    def plot_roc_curve(self, labels, predictions, title, ax, color='black'):
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        auc = roc_auc_score(labels, predictions)
        ax.plot(
            fpr,
            tpr,
            color=color,
            lw=2.0,
            label=f"AUC=%0.2f" % auc
        )
        ax.plot([0, 1], [0, 1], color="black", lw=1.0, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("1-Specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_title(title)
        ax.legend(loc="lower right")

    def plot_precision_recall(self, labels, predictions, title, ax, color='black'):
        precision, recall, thresholds = precision_recall_curve(
            labels, predictions)
        auc = average_precision_score(labels, predictions)
        ax.plot(
            recall,
            precision,
            color=color,
            lw=2.0,
            label=f"AUC=%0.2f" % auc
        )
        ax.plot([0, 1], [np.mean(labels), np.mean(labels)],
                color="black", lw=1.0, linestyle="--",
                label=f"Baseline AUC={round(np.mean(labels), 2)}")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="lower right")

    def calibration_curve_ci(self, y_true, y_prob, sample_weight=None,
                             n_bins=5):
        """
        Adapted from sklearn but allows but bootstraps CI and sample weights
        """
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        binids = np.searchsorted(bins[1:-1], y_prob)
        prob_trues, prob_preds = [], []
        inds = [i for i in range(len(y_prob))]
        df = pd.DataFrame(data={
            'bin_id': binids,
            'y_true': y_true,
            'sample_weight': sample_weight
        })
        for i in range(100):
            # Get bootstrapped sample (stratified by binid)
            df_boot = df.groupby('bin_id').sample(
                frac=1.0, replace=True).reset_index()
            if sample_weight is None:
                bin_sums = np.bincount(binids, weights=y_prob,
                                       minlength=len(bins))
                bin_true = np.bincount(df_boot['bin_id'].values,
                                       weights=df_boot['y_true'].values,
                                       minlength=len(bins))
                bin_total = np.bincount(binids, minlength=len(bins))

                nonzero = bin_total != 0
                prob_true = bin_true[nonzero] / bin_total[nonzero]
                prob_pred = bin_sums[nonzero] / bin_total[nonzero]

            else:
                bin_sums = np.bincount(binids, weights=y_prob,
                                       minlength=len(bins))
                bin_true = np.bincount(df_boot['bin_id'].values,
                                       weights=df_boot['y_true'].values *
                                       df_boot['sample_weight'].values,
                                       minlength=len(bins))
                bin_total = np.bincount(binids, minlength=len(bins))
                bin_total_true = np.bincount(
                    df_boot['bin_id'].values,
                    weights=df_boot['sample_weight'].values, minlength=len(
                        bins)
                )

                nonzero = bin_total != 0
                prob_true = bin_true[nonzero] / bin_total_true[nonzero]
                prob_pred = bin_sums[nonzero] / bin_total[nonzero]

            prob_trues.append(prob_true)
            prob_preds.append(prob_pred)

        return prob_trues, prob_preds

    def plot_calibration_curve(self, labels, predictions, title, ax, n_bins=5,
                               color='black', draw_baseline=True,
                               sample_weight=None):
        prob_trues, prob_preds = self.calibration_curve_ci(labels, predictions,
                                                           sample_weight=sample_weight,
                                                           n_bins=n_bins)
        prob_pred = np.mean(prob_preds, axis=0)
        prob_true = np.mean(prob_trues, axis=0)
        prob_true_lower = prob_true - np.percentile(prob_trues, 2.5, axis=0)
        prob_true_upper = np.percentile(prob_trues, 97.5, axis=0) - prob_true
        ax.scatter(
            prob_pred,
            prob_true,
            color=color
        )
        ax.errorbar(prob_pred,
                    prob_true,
                    np.vstack((prob_true_lower, prob_true_upper)),
                    color=color,
                    linestyle='')

        if draw_baseline:
            ax.plot([0, 1], [0, 1],
                    color="black", lw=1.0,
                    label=f"Perfectly Calibrated")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(title)
            ax.legend(loc="lower right")

    def bootstrap_metric(self, labels, predictions, metrics, iters=1000,
                         threshold=0.5):
        """
        Compute metric and 95% confidence interal
        """
        predictions = np.asarray(predictions)
        predicted_labels = np.asarray(
            [1 if p >= threshold else 0 for p in predictions])
        labels = np.asarray(labels)
        inds = [i for i in range(len(predictions))]
        values = {}
        actual_values = {}
        for i in range(iters):
            inds_b = np.random.choice(inds, size=len(inds), replace=True)
            l_b, p_b = labels[inds_b], predictions[inds_b]
            p_b_l = predicted_labels[inds_b]
            for m in metrics:
                if metrics[m].__name__ in THRESHOLD_DEPENDENT:
                    if m == 'Specificity':
                        values.setdefault(m, []).append(
                            metrics[m](l_b, p_b_l, pos_label=0))
                    else:
                        values.setdefault(m, []).append(metrics[m](l_b, p_b_l))
                else:
                    values.setdefault(m, []).append(metrics[m](l_b, p_b))

        for m in metrics:
            if metrics[m].__name__ in THRESHOLD_DEPENDENT:
                if m == 'Specificity':
                    actual_values[m] = metrics[m](
                        labels, predicted_labels, pos_label=0)
                else:
                    actual_values[m] = metrics[m](labels, predicted_labels)
            else:
                actual_values[m] = metrics[m](labels, predictions)

        results = {}
        for v in values:
            mean = '{:.2f}'.format(round(actual_values[v], 2))
            upper = '{:.2f}'.format(round(np.percentile(values[v], 97.5), 2))
            lower = '{:.2f}'.format(round(np.percentile(values[v], 2.5), 2))
            results[v] = f"{mean} [{lower}, {upper}]"

        return results

class BinaryEvaluatorByTime(BinaryEvaluator):
    def __init__(self, outdir):
        self.outdir = outdir
        self.metrics = {
            # 'Accuracy': accuracy_score,
            # 'Sensitivity': recall_score,
            # 'Specificity': recall_score,
            # 'Precision': precision_score,
            'AUROC': roc_auc_score,
            # 'Average precision': average_precision_score
        }
    def __call__(self, labels, predictions, index_times):
        """
        Override in child classes to include functionality to pull labels and
        predictions from some outside source (ex: cosmos db)
        """
        self.get_performance_artifacts_by_time(labels=labels, predictions=predictions, index_times=index_times)

    def get_performance_artifacts_by_time(self, labels, predictions, index_times):
        #Get years
        years = index_times.dt.year
        years_in_test = list(set(years))
        years_in_test.sort()
        inds = [i for i in range(len(years))]

        #Group by year
        labels_in_year = {}
        preds_in_year = {}
        for year in years_in_test:
            year_idx_mask = [1 if years.iloc[i] == year else 0 for i in range(len(years))]
            labels_in_year[year] = []
            preds_in_year[year] = []
            for i in range(len(year_idx_mask)):
                if year_idx_mask[i] == 1:
                    labels_in_year[year].append(labels.iloc[i])
                    preds_in_year[year].append(predictions.iloc[i])
                assert len(labels_in_year) == len(preds_in_year)

        # Bootstrap metric by time
        results_by_year = {}
        for year in years_in_test:
            result = self.bootstrap_metric(labels_in_year[year], preds_in_year[year], self.metrics)
            results_by_year[year] = result

        with open(os.path.join(self.outdir, "auc_ci.json"), "w") as fp:
            json.dump(results_by_year, fp)

        fig, ax = plt.subplots(nrows=1, ncols=1)

        # Plot AUC By Year
        self.plot_auc_by_year(labels_in_year, preds_in_year, results_by_year, title="AUC By Year", ax=ax)
        plt.savefig(os.path.join(self.outdir, 'auc_by_year.png'),
                    bbox_inches='tight',
                    dpi=300)

    def plot_auc_by_year(self, labels_by_year, preds_by_year, bootstrap_by_year, title, ax, color='black'):
        years = list(labels_by_year.keys())
        years.sort()

        aucs = []
        lower_vals = []
        upper_vals = []     
        for year in years:
            #Get AUC score
            auc_in_year = roc_auc_score(labels_by_year[year], preds_by_year[year])
            aucs.append(auc_in_year)
            #Get CIs
            errors = bootstrap_by_year[year]['AUROC']
            l_index = errors.index("[")
            r_index = errors.index("]")
            lower = float(errors[l_index + 1: l_index + 5])
            upper = float(errors[r_index - 4: r_index])
            lower_vals.append(lower)
            upper_vals.append(upper)
        ax.plot(
            years,
            aucs,
            color=color,
            lw=2.0,
            label=f"Max AUC: {max(aucs):.2f}\nMin AUC: {min(aucs):.2f}"
        )
        errors = np.stack([np.subtract(aucs,lower_vals), np.subtract(upper_vals, aucs)])
        ax.errorbar(
            years,
            aucs, 
            yerr = errors,
            color=color
        )
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Year")
        ax.set_ylabel("AUC")
        ax.set_title(title)
        leg = ax.legend(loc="lower right", handlelength=0, handletextpad=0, fancybox=True)
        for item in leg.legendHandles:
            item.set_visible(False)
        max_auc = max(aucs)
        max_drop = max(aucs) - min(aucs)
        
        #Save values
        values = {}
        values['max_auc'] = max(aucs)
        values['max_drop'] = max(aucs) - min(aucs)
        with open(os.path.join(self.outdir, "auc_max.json"), 'w') as fp:
            json.dump(values, fp)