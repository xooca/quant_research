import data.utils.duckdb_utils as ddu
import importlib
import data.utils.data_utils as du

class define_pipeline:
    def __init__(self, 
                 specification_name,
                 master_config_path, 
                 master_config_name,
                 db_connection,
                 database_path=None, 
                 train_feature_table = None,
                 train_feature_selection_table=None,
                 train_feature_info_table=None,
                 train_training_info_table=None,
                 train_tuning_info_table=None,
                 ignore_column = [],
                 verbose=True):

        self.specification_name = specification_name
        du.print_log(f"Specification Name is file is {self.specification_name}")
        self.specification = importlib.import_module(f"{self.specification_name}")
        self.pipeline = self.specification.model(master_config_path=master_config_path, 
                                                    master_config_name=master_config_name,
                                                    db_connection=db_connection,
                                                    database_path=database_path, 
                                                    train_feature_table = train_feature_table,
                                                    train_feature_selection_table=train_feature_selection_table,
                                                    train_feature_info_table=train_feature_info_table,
                                                    train_training_info_table=train_training_info_table,
                                                    train_tuning_info_table=train_tuning_info_table,
                                                    ignore_column = ignore_column,
                                                    verbose=verbose)

    def run_training_pipelines(self,
                                model_fit_params,
                                prob_theshold_list=None,
                                model_params=None,
                                only_run_for_label = [],
                                forced_labels = [],
                                feature_selection_method='featurewiz',
                                filter_out_cols=None):
        self.pipeline.train_all_labels(model_fit_params=model_fit_params,
                                                prob_theshold_list=prob_theshold_list,
                                                model_params=model_params,
                                                only_run_for_label = only_run_for_label,
                                                forced_labels = forced_labels,
                                                feature_selection_method=feature_selection_method,
                                                filter_out_cols=filter_out_cols)
        
    def run_tuning_pipelines(self,only_run_for_label = [],
                         forced_labels = [],
                         feature_selection_method='featurewiz',
                         limit=2
                            ):
        self.pipeline.tune_all_labels(only_run_for_label = only_run_for_label,
                                        forced_labels = forced_labels,
                                        feature_selection_method=feature_selection_method,
                                        limit=limit)
