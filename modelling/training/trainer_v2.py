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
                 train_running_info_table=None,
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
                                                                            train_running_info_table=train_running_info_table,
                                                                            verbose=verbose)

    def run_pipelines(self,model_params,model_fit_params,prob_theshold_list=None,label_name = None,feature_selection_method='featurewiz'):
        self.training_pipeline.train_all_labels(model_params,model_fit_params,
                                                prob_theshold_list=prob_theshold_list,
                                                label_name = label_name,
                                                feature_selection_method=feature_selection_method)