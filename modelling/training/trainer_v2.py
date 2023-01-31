import data.utils.duckdb_utils as ddu
import importlib
import data.utils.data_utils as du

class execute_training_pipeline:
    def __init__(self, 
                 training_spec_name,
                 master_config_path, 
                 master_config_name,
                 db_connection,
                 database_path=None, 
                 train_feature_table = None,
                 train_feature_selection_table=None,
                 train_feature_info_table=None,
                 train_training_info_table=None,
                 verbose=True):

        
        
        self.training_spec_name = training_spec_name
        du.print_log(f"Training spec file is {self.training_spec_name}")
        self.training_spec = importlib.import_module(f"{self.training_spec_name}")
        self.training_pipeline = self.training_spec.model(master_config_path=master_config_path, 
                                                                            master_config_name=master_config_name,
                                                                            db_connection=db_connection,
                                                                            database_path=database_path, 
                                                                            train_feature_table = train_feature_table,
                                                                            train_feature_selection_table=train_feature_selection_table,
                                                                            train_feature_info_table=train_feature_info_table,
                                                                            train_training_info_table=train_training_info_table,
                                                                            verbose=verbose)

    def run_pipelines(self,model_params,model_fit_params,prob_theshold_list=None,forced_label_name = None,feature_selection_method='featurewiz'):
        self.training_pipeline.train_all_labels(model_params,model_fit_params,
                                                prob_theshold_list=prob_theshold_list,
                                                forced_label_name = forced_label_name,
                                                feature_selection_method=feature_selection_method)
        

class execute_tuning_pipeline:
    def __init__(self, 
                 tuning_spec_name,
                 master_config_path, 
                 master_config_name,
                 db_connection,
                 database_path=None, 
                 train_feature_table = None,
                 train_feature_selection_table=None,
                 train_tuning_info_table=None,
                 verbose=True):

    
        self.tuning_spec_name = tuning_spec_name
        du.print_log(f"Tuning spec file is {self.tuning_spec_name}")
        self.tuning_spec = importlib.import_module(f"{self.tuning_spec_name}")
        self.tuning_pipeline = self.tuning_spec.tuner_model(master_config_path=master_config_path, 
                                                                            master_config_name=master_config_name,
                                                                            db_connection=db_connection,
                                                                            database_path=database_path, 
                                                                            train_feature_table = train_feature_table,
                                                                            train_feature_selection_table=train_feature_selection_table,
                                                                            train_tuning_info_table=train_tuning_info_table,
                                                                            verbose=verbose)

    def run_pipelines(self,forced_label_name,
                      feature_selection_method,
                      force_tuning_labels=None,
                      limit = 2,
                      ):
        self.tuning_pipeline.tune_all_labels(forced_label_name = forced_label_name,
                                             feature_selection_method=feature_selection_method,
                                             force_tuning_labels=force_tuning_labels,limit=limit)