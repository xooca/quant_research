import pandas as pd
#import data_engine as de
from pycaret.classification import setup,compare_models,create_model,evaluate_model,plot_model,tune_model,blend_models,stack_models,finalize_model,save_model,load_model,predict_model
from pycaret.utils import check_metric
import sklearn.metrics as mt
import numpy as np
from modelling.model_specs.base import BaseModel
import data.data_utils as du
import os
import omegaconf
from config.common.config import Config
import lightgbm as lgb
import data.utils.duckdb_utils as ddu
import numpy as np
import optuna.integration.lightgbm as opt_lgb
from lightgbm import early_stopping
from lightgbm import log_evaluation
import lightgbm as lgb
import optuna
from shaphypetune import BoostSearch, BoostBoruta, BoostRFE, BoostRFA
import featurewiz


class modelling(BaseModel):
    def __init__(self,master_config_path,master_config_name,model_spec_name=None):
        BaseModel.__init__(self,master_config_path,master_config_name)
        if model_spec_name is not None:
            self.model_spec_name = model_spec_name
        print(f"Model spec is {self.model_spec_name}")
        self.define_parameters_custom()

    def define_parameters_custom(self):
        self.setup = self.config.model.trainer.setup
        self.tune_args = self.config.model.trainer.tune_model
        self.blend_args = self.config.model.trainer.blend_model
        self.stack_args = self.config.model.trainer.stack_model
        self.check_metric = self.config.model.trainer.check_metric
        self.create_model_args = self.config.model.trainer.create_model

    def initial_setup(self): 
        self.setup = dict(self.setup)
        self.setup['data'] = self.train
        if self.valid is not None:
            self.setup['test_data'] = self.valid
            du.print_log(f"Valid dataset is defined",self.using_print)
        self.experiment_setup = setup(**self.setup)
        self.model_metadata.update({'model_setup_args':self.setup})
        self.save_model_artifacts()
    
    def get_train_val_data(self,db_connection,feature_names,label_name,table_name,
                           train_combination,validation_combination,if_return_df = False):
        train_data = ddu.load_table_df(db_connection,column_names=feature_names,table_name=table_name,filter=filter)
        train_data_label = ddu.load_table_df(db_connection,column_names=label_name,table_name=table_name,filter=train_combination)
        validation_data = ddu.load_table_df(db_connection,column_names=feature_names,table_name=table_name,filter=validation_combination)
        validation_data_label = ddu.load_table_df(db_connection,column_names=label_name,table_name=table_name,filter=validation_combination)
        print('Training Data Shape: ', train_data.shape)
        print('Testing Data Shape: ', validation_data.shape)
        if if_return_df is False:
            train_data = np.array(train_data)
            train_data_label = np.array(train_data_label)
            validation_data = np.array(validation_data)
            validation_data_label = np.array(validation_data_label)
        return train_data,train_data_label,validation_data,validation_data_label
            
    def model_train(self,db_connection,fold_combinations,feature_names,feature_table_name,label_name,model_params,model_fit_params):
        self.model = lgb.LGBMClassifier(**model_params)
        for train_combination,validation_combination in fold_combinations:
            train_data,train_data_label,validation_data,validation_data_label= self.get_train_val_data(db_connection=db_connection,
                                                                                                    feature_names=feature_names,
                                                                                                    label_name=label_name,
                                                                                                    table_name=feature_table_name,
                                                                                                    train_combination=train_combination,
                                                                                                    validation_combination=validation_combination)
            
            model_fit_params.update({'X':train_data})
            model_fit_params.update({'y':train_data_label})
            model_fit_params.update({'eval_set':[(validation_data, validation_data_label), (train_data, train_data_label)]})
            model_fit_params.update({'eval_names':['valid','train']})
            self.model.fit(**model_fit_params)
        print(self.model.best_score_)
    
    def model_prediction(self,features_names,predict_mode='proba',test_data=None,db_connection=None,test_table_name=None,table_filter=None,model=None):
        if test_table_name is not None and db_connection is not None:
            test_data = ddu.load_table_df(db_connection=db_connection,
                                       column_names = features_names,
                                       table_name=test_table_name,
                                       filter=table_filter)
            test_data = np.array(test_data[[features_names]])
        if isinstance(test_data,pd.DataFrame):
            #test_data = test_data[[features_name]]
            test_data = np.array(test_data[[features_names]])
        if model is None:
            if predict_mode == 'proba':
                preds = self.model.predict_proba(test_data)
            else:
                preds = self.model.predict(test_data)
        else:
            if predict_mode == 'proba':
                preds = model.predict_proba(test_data)
            else:
                preds = model.predict(test_data)     
        return preds      
            
    def model_evaluation(self,features_name,label_name=None,actual_labels=None,test_data=None,db_connection=None,test_table_name=None,table_filter=None,model=None):
        metric_dic={}
        preds_proba = self.model_prediction(features_name,predict_mode='proba',test_data=test_data,db_connection=db_connection,test_table_name=test_table_name,table_filter=table_filter,model=model)
        preds_predict = self.model_prediction(features_name,predict_mode='predict',test_data=test_data,db_connection=db_connection,test_table_name=test_table_name,table_filter=table_filter,model=model)
        if actual_labels is None:
            test_data = ddu.load_table_df(db_connection=db_connection,
                                           column_name = features_name,
                                           table_name=test_table_name,
                                           filter=table_filter)
            actual_labels = ddu.load_table_df(db_connection=db_connection,
                                              column_name = label_name,
                                              table_name=test_table_name,
                                              filter=table_filter)
        classification_report = mt.classification_report(actual_labels,preds_predict)
        metric_dic.update({'classification_report':classification_report})
        balanced_accuracy_score = mt.balanced_accuracy_score(actual_labels,preds_predict)
        metric_dic.update({'balanced_accuracy_score':balanced_accuracy_score})
        accuracy_score = mt.accuracy_score(actual_labels,preds_predict)
        metric_dic.update({'accuracy_score':accuracy_score})
        top_k_accuracy_score = mt.top_k_accuracy_score(actual_labels,preds_predict)
        metric_dic.update({'top_k_accuracy_score':top_k_accuracy_score})
        roc_auc_score = mt.roc_auc_score(actual_labels,preds_proba,multi_class='ovr')
        metric_dic.update({'roc_auc_score':roc_auc_score})
        f1_score = mt.f1_score(actual_labels,preds_proba)
        metric_dic.update({'f1_score':f1_score})
        auc = mt.auc(actual_labels,preds_proba)
        metric_dic.update({'auc':auc})
        precision_score = mt.precision_score(actual_labels,preds_proba)
        metric_dic.update({'precision_score':precision_score})
        recall_score = mt.recall_score(actual_labels,preds_proba)
        metric_dic.update({'recall_score':recall_score})
        return metric_dic

    def model_tuning_fn(self,trial,train_combination,validation_combination):
        train_data,train_data_label,validation_data,validation_data_label= self.get_train_val_data(db_connection=self.db_connection,
                                                                                                feature_names=self.feature_names,
                                                                                                label_name=self.label_name,
                                                                                                table_name=self.table_name,
                                                                                                train_combination=train_combination,
                                                                                                validation_combination=validation_combination)
        
        train_data_lgb = lgb.Dataset(train_data, label=train_data_label)
        validation_data_lgb = lgb.Dataset(validation_data, label=validation_data_label)

        if self.tuning_params is None:
            self.tuning_params = {
            #'objective': 'binary',
            #'metric': 'binary_logloss',
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
             }
                    
            unique_items, label_cnt = np.unique(train_data_label, return_counts=True)
            if label_cnt > 2:
                self.tuning_params.update({'objective':'multiclass'})
                self.tuning_params.update({'metric':'multiclass'})
            else:
                self.tuning_params.update({'objective':'binary'})
                self.tuning_params.update({'metric':'binary_logloss'})

    
        model = lgb.train(self.tuning_params, train_data_lgb,valid_sets=[train_data_lgb, validation_data_lgb],callbacks=[early_stopping(100), log_evaluation(100)])
        prediction = np.rint(model.predict(validation_data, num_iteration=model.best_iteration))
        accuracy = mt.accuracy_score(validation_data_label, prediction)
        best_params = model.params
        print("Best params:", best_params)
        print("  Accuracy = {}".format(accuracy))
        print("  Params: ")
        for key, value in best_params.items():
            print("    {}: {}".format(key, value))
        return accuracy
    
    def model_tuning(self,tuning_params,n_trials,train_combination,validation_combination,direction='maximize'):
        self.tuning_params = tuning_params
        wrapped_func = lambda trial: self.model_tuning_fn(trial, train_combination,validation_combination)

        self.study = optuna.create_study(direction=direction)
        self.study.optimize(wrapped_func, n_trials=n_trials)
        
        print('Number of finished trials:', len(self.study.trials))
        print('Best trial:', self.study.best_trial.params)
        
    def feature_selection(self,method,train_combination,validation_combination,feature_selection_param):
        train_data,train_data_label,validation_data,validation_data_label= self.get_train_val_data(db_connection=self.db_connection,
                                                                                                feature_names=self.feature_names,
                                                                                                label_name=self.label_name,
                                                                                                table_name=self.table_name,
                                                                                                train_combination=train_combination,
                                                                                                validation_combination=validation_combination,
                                                                                                if_return_df=True)

        if method == 'BoostBoruta':
            ret_obj = BoostBoruta(**feature_selection_param)
            ret_obj.fit(train_data, 
                      train_data_label, 
                      eval_set=[(validation_data, validation_data_label)], 
                      early_stopping_rounds=6, verbose=0)
        elif method == 'BoostRFE':
            ret_obj = BoostRFE(**feature_selection_param)
            ret_obj.fit(train_data, 
                      train_data_label, 
                      eval_set=[(validation_data, validation_data_label)], 
                      early_stopping_rounds=6, verbose=0) 
        elif method == 'BoostRFA':
            ret_obj = BoostRFA(**feature_selection_param)
            ret_obj.fit(train_data, 
                      train_data_label, 
                      eval_set=[(validation_data, validation_data_label)], 
                      early_stopping_rounds=6, verbose=0)  
        elif method == 'featurewiz':
            ret_obj = featurewiz(train_data, train_data_label)
            
        return ret_obj
                    
    def model_saving(self):
        path = None
        try:
            path = f"{self.model_save_path}_{self.estimator}"
            save_model(self.final_model, path)
            self.model_metadata.update({'model_save_status':'success'})
        except Exception as e:
            du.print_log(f"Unable to save model to {path}",self.using_print)
            du.print_log(f"Error encountered is {e}",self.using_print)
            self.model_metadata.update({'model_save_status':'failed'})
        self.model_metadata.update({'model_path':path})
        self.save_model_artifacts()


    def save_model_artifacts(self):
        du.save_object(object_path=self.model_metadata_save_path,obj=self.model_metadata)
     
    def trainer(self):
        #self.model_spec_obj = self.model_spec.modelling(self.model_config)

        #self.initial_setup()
        #du.print_log("******** Setup Completed ************",self.using_print)
        self.model_creation()
        du.print_log("******** Model Creation Completed ************",self.using_print)
        self.model_tuning(from_compare=False)
        du.print_log("******** Model Tuning Completed ************",self.using_print)
        self.model_finalization()
        du.print_log("******** Model Finalization Completed ************",self.using_print)
        self.model_saving()
        du.print_log("******** Model Saving Completed ************",self.using_print)
        self.model_evaluation()
        du.print_log("******** Model Evaluation Completed ************",self.using_print)
        self.model_prediction(self.test,check_metric_flag=True)
        du.print_log("******** Model Prediction Completed ************",self.using_print)
        #self.save_model_artifacts()
        #du.print_log("*************** Saved Metadata *****************",self.using_print)