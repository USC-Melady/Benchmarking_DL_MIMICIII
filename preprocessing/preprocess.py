from argparse import ArgumentParser

from preprocessing.steps.createAdmissionList import createAdmissionList
from preprocessing.steps.getItemIdList import getItemIdList
from preprocessing.steps.filterItemId_input import filterItemId_input
from preprocessing.steps.filterItemId_output import filterItemId_output
from preprocessing.steps.filterItemId_chart import filterItemId_chart
from preprocessing.steps.filterItemId_lab import filterItemId_lab
from preprocessing.steps.filterItemId_microbio import filterItemId_microbio
from preprocessing.steps.filterItemId_prescript import filterItemId_prescript
from preprocessing.steps.processing import processing
from preprocessing.steps.collect_mortality_labels import collect_mortality_labels
from preprocessing.steps.getValidDataset import getValidDataset
from preprocessing.steps.run_necessary_sqls import run_necessary_sqls
from preprocessing.steps.get_17_features_processed import get_17_features_processed
from preprocessing.steps.get_17_features_raw import get_17_features_raw
from preprocessing.steps.get_99plus_features_raw import get_99plus_features_raw
from preprocessing.steps.get_time_series_sample_17_features_processed import get_time_series_sample_17_features_processed
from preprocessing.steps.get_time_series_sample_17_features_raw import get_time_series_sample_17_features_raw
from preprocessing.steps.get_time_series_sample_99plus_features_raw import get_time_series_sample_99plus_features_raw
from preprocessing.steps.get_avg_17_features_processed import get_avg_17_features_processed
from preprocessing.steps.get_avg_17_features_raw import get_avg_17_features_raw
from preprocessing.steps.get_avg_99plus_features_raw import get_avg_99plus_features_raw


def main():
    parser = ArgumentParser()
    parser.add_argument('--cachedir', default='data')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    # # select admissions and all features
    createAdmissionList(args)
    getItemIdList(args)
    filterItemId_input(args)
    filterItemId_output(args)
    filterItemId_chart(args)
    filterItemId_lab(args)
    filterItemId_microbio(args)
    filterItemId_prescript(args)
    processing(args)
    collect_mortality_labels(args)
    getValidDataset(args)

    # # extract 17 processed features, 17 raw features and 99+ raw features
    run_necessary_sqls(args)
    get_17_features_processed(args)
    get_17_features_raw(args)
    get_99plus_features_raw(args)

    # # generate time series
    get_time_series_sample_17_features_processed(args)
    get_time_series_sample_17_features_raw(args)
    get_time_series_sample_99plus_features_raw(args)

    # generate non-temporal features for superlearner
    get_avg_17_features_processed(args)
    get_avg_17_features_raw(args)
    get_avg_99plus_features_raw(args)

if __name__ == '__main__':
    main()
