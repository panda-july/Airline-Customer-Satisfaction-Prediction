# %% [markdown]
# # Define Pipeline

# %% [markdown]
# * [How To Write Clean And Scalable Code With Custom Transformers & Sklearn Pipelines](https://medium.com/@benlc77/how-to-write-clean-and-scalable-code-with-custom-transformers-sklearn-pipelines-ecb8e53fe110)
# * [Missing value imputation using Sklearn pipelines](https://marloz.github.io/projects/sklearn/pipeline/missing/preprocessing/2020/03/20/sklearn-pipelines-missing-values.html)
# * [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
# * [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)

# %% [code]
import numpy as np
import pandas as pd
#from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder,FunctionTransformer

class CustomRename(BaseEstimator, TransformerMixin):
    """
        Rename column names
    """
    def __init__(self, update_col_names=None):
        self.update_col_names = update_col_names
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        org_col_list=X.columns
        if self.update_col_names is not None:
            rename_dic=dict(zip(org_col_list, self.update_col_names))
            new_X=X.rename(columns=rename_dic)
        else:
            #replace column name from to
            replace_dict={
                " ": "_",
                "-": "_",
                "&": "_",
                "(": "",
                ")": "",
                "[": "",
                "]": "",
                ",": "_",
                "<": "lt",
                ">=": "gte",
                "/": "_"
            }
            new_col_list=[]
            for org_col in org_col_list:
                #to lower case
                new_col=org_col.lower()
                for replace_from,replace_to in zip(replace_dict.keys(),replace_dict.values()):
                    new_col=new_col.replace(replace_from, replace_to)
                new_col_list.append(new_col)
            rename_dic=dict(zip(org_col_list, new_col_list))
            new_X=X.rename(columns=rename_dic)
        return new_X


#https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
class CustomImputer(BaseEstimator, TransformerMixin):
    """
        Clean dataframe, handle missing values
    """
    def __init__(self, missing_values = np.nan, 
                        strategy ='mean'):
        """
            missing_values: The placeholder for the missing values(pass it to SimpleImputer)
            strategy: The imputation strategy, 'drop_row' | 'drop_col' | 'mean' | 'median' | 'most_frequent' | 'constant'
                     drop_row: delete rows which have missing value
                     drop_col: delete columns which have missing value
                     others: pass it to SimpleImputer
        """
        self.missing_values = missing_values
        self.strategy = strategy
    def fit(self, X, y = None):
        if self.strategy == 'drop_row':
            pass
        elif self.strategy == 'drop_col':
            pass
        else:
            self.simple_imputer=SimpleImputer(missing_values=self.missing_values, strategy=self.strategy)
            self.simple_imputer=self.simple_imputer.fit(X)
        return self
    def transform(self, X):
        if self.strategy == 'drop_row':
            result=X.dropna(axis='index')
        elif self.strategy == 'drop_col':
            result=X.dropna(axis='columns')
        else:
            result=self.simple_imputer.transform(X)
        return result


class CustomFeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, variables = None):
        self.variables = variables
    def fit(self, X, y =None):
        return self
    def transform(self, X):
        # source: https://www.kaggle.com/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition#Feature-Engineering
        new_X = X.copy()
        #convert delay time to hour
        new_X["departure_delay_in_hours"]=round(new_X["departure_delay_in_minutes"]/60,0)
        #Departure Delay in Minutes
        new_X['departure_delay_class'] = new_X['departure_delay_in_minutes'].apply(self.split_delay)

        #arrival_delay_in_minutes has missing values, in some case it might be dropped
        if "arrival_delay_in_minutes" in X.columns.to_list():
            new_X["arrival_delay_in_hours"]=round(new_X["arrival_delay_in_minutes"]/60,0)
            #Arrival Delay in Minutes
            new_X['arrival_delay_class'] = new_X['arrival_delay_in_minutes'].apply(self.split_delay)
        
        #split_flight_distance
        new_X['flight_distance_range'] = new_X['flight_distance'].apply(self.split_flight_distance_to_categorical)
        #transform distance to airline distance classification
        new_X['flight_distance_class'] = new_X['flight_distance'].apply(self.split_flight_distance)
        #Transform Age
        #transform Age to range
        new_X['age_range'] = new_X['age'].apply(self.split_age_to_categorical)
        #transform Age to airline age classification
        new_X['age_class_by_airline'] = new_X['age'].apply(self.split_age_to_airline_class)
        #Transform Rating
        new_X['seat_comfort_class'] = new_X['seat_comfort'].apply(self.split_rating_to_range)
        new_X['departure_arrival_time_convenient_class'] = new_X['departure_arrival_time_convenient'].apply(self.split_rating_to_range)
        new_X['food_and_drink_class'] = new_X['food_and_drink'].apply(self.split_rating_to_range)
        new_X['gate_location_class'] = new_X['gate_location'].apply(self.split_rating_to_range)
        new_X['inflight_wifi_service_class'] = new_X['inflight_wifi_service'].apply(self.split_rating_to_range)
        new_X['inflight_entertainment_class'] = new_X['inflight_entertainment'].apply(self.split_rating_to_range)
        new_X['online_support_class'] = new_X['online_support'].apply(self.split_rating_to_range)
        new_X['ease_of_online_booking_class'] = new_X['ease_of_online_booking'].apply(self.split_rating_to_range)
        new_X['on_board_service_class'] = new_X['on_board_service'].apply(self.split_rating_to_range)
        new_X['leg_room_service_class'] = new_X['leg_room_service'].apply(self.split_rating_to_range)
        new_X['baggage_handling_class'] = new_X['baggage_handling'].apply(self.split_rating_to_range)
        new_X['checkin_service_class'] = new_X['checkin_service'].apply(self.split_rating_to_range)
        new_X['cleanliness_class'] = new_X['cleanliness'].apply(self.split_rating_to_range)
        new_X['online_boarding_class'] = new_X['online_boarding'].apply(self.split_rating_to_range)
        #Create Class_M
        #merge Eco and Eco Plus
        class_merge_map={'Eco': 'Eco&EcoPlus', 'Eco Plus': 'Eco&EcoPlus', 'Business': 'Business'}
        new_X['class_m']=new_X['class'].map(class_merge_map)
        return new_X


    def split_delay(self,x):
        '''
            Split delay time to delay groups
            T<15 minutes: No delay
            15<=T<60 minutes: Minor delay
            60<=T<240 minutes: Moderate delay
            T=>240 minutes: Significant delay
        '''
        if (0<=x)&(x < 15):
            delay_class = 'No Delay'
        elif x < 60:
            delay_class = 'Minor Delay'
        elif x < 240:
            delay_class = 'Moderate Delay'
        else:
            delay_class = 'Significant Delay'
        return delay_class

    #transform distance to distance range
    def split_flight_distance_to_categorical(self,x):
        if x < 1000:
            flight_class = '<1K'
        elif x < 3000:
            flight_class = '[1K,3K)'
        elif x < 4000:
            flight_class = '[3K,4K)'
        else:
            flight_class = '>=4K'
        return flight_class
        
    #transform distance to airline distance classification
    def split_flight_distance(self,x):
        if x <= 700:
            flight_class = 'Short-Haul'
        elif x <= 3000:
            flight_class = 'Medium-Haul'
        else:
            flight_class = 'Long-Haul'
        return flight_class

    #transform Age to range
    def split_age_to_categorical(self,x):
        if x < 10:
            Age_range = '<10'
        elif x < 20:
            Age_range = '[10,20)'
        elif x < 30:
            Age_range = '[20,30)'
        elif x < 40:
            Age_range = '[30,40)'
        elif x < 50:
            Age_range = '[40,50)'
        elif x < 60:
            Age_range = '[50,60)'
        elif x < 70:
            Age_range = '[60,70)'
        else:
            Age_range = '>=70'
        return Age_range

    #transform Age to airline age classification
    def split_age_to_airline_class(self,x):
        if x < 2:
            age_airline_class = 'Infants'
        elif x < 12:
            age_airline_class = 'Children'
        elif x < 18:
            age_airline_class = 'Young-Adult'
        elif x < 65:
            age_airline_class = 'Adult'
        else:
            age_airline_class = 'Senior'
        return age_airline_class
        
    #transform rating score to rating classification
    def split_rating_to_range(self,x):
        '''
            0<=rating<2: Low
            2<=rating<4: Median
            4<=rating<6: High
        '''
        if x < 2:
            rating_class = 'Low'
        elif x < 4:
            rating_class = 'Median'
        else:
            rating_class = 'High'
        return rating_class

  
class CustomEncoder(BaseEstimator, TransformerMixin):
    """
        Encode dataframe
    """
    def __init__(self, ordinal_col_list = None, 
                dummy_col_list =None):
        """
            ordinal_col_list: ordinal encoding column name list
            dummy_col_list: dummy encoding column name list
        """
        self.ordinal_col_list = ordinal_col_list
        self.dummy_col_list = dummy_col_list
        #column names after dummy encoding(e.g. ColumnA-> CoulmnA_dummy_Group1, CoulmnA_dummy_Group2)
        self.dummy_encoded_cols=[]
        
    def fit(self, X, y = None):
        #check duplicated columns
        self.chk_duplicated_cols(self.ordinal_col_list,self.dummy_col_list)
        self.get_encode_columns(X,self.ordinal_col_list,self.dummy_col_list)
        if y is not None:
            set_exclude_col=set(self.exclude_col_list)
            set_exclude_col=set_exclude_col.union([y.name])
            self.exclude_col_list=list(set_exclude_col)
        return self
        
    def transform(self, X):
        ordinal_encode_result=self.ordinal_encode(X)
        dummy_encode_result=self.dummy_encode(X)
        if (len(self.encode_ordinal_cols)>0) & (len(self.encode_dummy_cols)>0):
            ordinal_encode_merge=ordinal_encode_result.drop(columns=self.encode_dummy_cols)
            dummy_encode_merge=dummy_encode_result.drop(columns=self.no_encode_cols+self.encode_ordinal_cols)
            result=pd.merge(ordinal_encode_merge, dummy_encode_merge, how="inner",left_index=True, right_index=True)
        elif len(self.encode_ordinal_cols)>0:
            result=ordinal_encode_result
        else:
            result=dummy_encode_result
        return result
    
    def chk_duplicated_cols(self,ordinal_col_list,dummy_col_list):
        """
            Check if ordinal_col_list and dummy_col_list have duplicated columns
        """
        if (ordinal_col_list is None) | (dummy_col_list is None):
            pass
        elif (len(ordinal_col_list)==0) | (len(dummy_col_list)==0):
            pass
        else:
            if len(ordinal_col_list)+len(dummy_col_list)!=len(set(self.ordinal_col_list+self.dummy_col_list)):
                raise Exception("Duplicated elements in ordinal_col_list and dummy_col_list")

    def get_encode_columns(self,X,ordinal_col_list,dummy_col_list):
        target_ordinal_cols=[]
        target_dummy_cols=[]
        all_cols=X.columns
        #if columns defined in ordinal_col_list also exist in dataframe, these columns will be ordinal encoding
        if (self.ordinal_col_list is not None) and len(self.ordinal_col_list)>0:
            for col in all_cols:
                if col in self.ordinal_col_list:
                    target_ordinal_cols.append(col)
        #if columns defined in dummy_col_list also exist in dataframe, these columns will be dummy encoding
        if (self.dummy_col_list is not None) and len(self.dummy_col_list)>0:
            for col in all_cols:
                if col in self.dummy_col_list:
                    target_dummy_cols.append(col)
        self.encode_ordinal_cols=target_ordinal_cols
        self.encode_dummy_cols=target_dummy_cols
        #columns need to be encoded
        set_ordinal_dummy=set(target_ordinal_cols+target_dummy_cols)
        #columns will not be encoded
        no_encode_cols=list(set(all_cols)-set_ordinal_dummy)
        self.no_encode_cols=no_encode_cols

    def dummy_encode(self, X):
        """
            Onehot encoding features exsit in dummy_col_list
        """
        dummy_encoded_cols=[]
        #rating item columns
        rating_item_list=['seat_comfort',
           'departure_arrival_time_convenient', 'food_and_drink',
           'gate_location', 'inflight_wifi_service',
           'inflight_entertainment', 'online_support',
           'ease_of_online_booking', 'on_board_service',
           'leg_room_service', 'baggage_handling',
           'checkin_service', 'cleanliness', 'online_boarding']
        
        #print(self.encode_dummy_cols)
        if len(self.encode_dummy_cols)>0:
            #if rating items need to dummy encoding, convert data type to string
            for col in self.encode_dummy_cols:
                if col in rating_item_list:
                    X[col]=X[col].astype(str)
            dummy_trans_X=pd.get_dummies(X, columns=self.encode_dummy_cols, prefix_sep='_dummy_',dtype=int,drop_first=True)
            #rename columns
            custom_rename = CustomRename()
            custom_rename=custom_rename.fit(dummy_trans_X)
            dummy_trans_X = custom_rename.transform(dummy_trans_X)
            #get dummy feature names
            for col in dummy_trans_X.columns:
                if '_dummy_' in col:
                   dummy_encoded_cols.append(col) 
            self.dummy_encoded_cols=dummy_encoded_cols
            return dummy_trans_X
    def ordinal_encode(self,X):
        """
            Ordinal encoding features exsit in ordinal_col_list
        """
        new_X=X.copy()
        rating_class_list=['seat_comfort_class',
           'departure_arrival_time_convenient_class', 'food_and_drink_class',
           'gate_location_class', 'inflight_wifi_service_class',
           'inflight_entertainment_class', 'online_support_class',
           'ease_of_online_booking_class', 'on_board_service_class',
           'leg_room_service_class', 'baggage_handling_class',
           'checkin_service_class', 'cleanliness_class', 'online_boarding_class']
        if len(self.encode_ordinal_cols)>0:
            for col in self.encode_ordinal_cols:
                if col == 'class':
                    encode_order=['Eco','Eco Plus','Business']
                elif col == 'class_m':
                    encode_order=['Eco&EcoPlus','Business']
                elif col == 'age_range':
                    encode_order=['<10','[10,20)','[20,30)','[30,40)','[40,50)','[50,60)','[60,70)','>=70']
                elif col == 'age_class_by_airline':
                    encode_order=['Infants','Children','Young-Adult','Adult','Senior']
                elif col == 'flight_distance_range':
                    encode_order=['<1K','[1K,3K)','[3K,4K)','>=4K']
                elif col == 'flight_distance_class':
                    encode_order=['Short-Haul','Medium-Haul','Long-Haul']
                elif col == 'departure_delay_class':
                    encode_order=['No Delay','Minor Delay','Moderate Delay','Significant Delay']
                elif col == 'satisfaction':
                    encode_order=['dissatisfied','satisfied']
                elif col in rating_class_list:
                    encode_order=['Low','Median','High']
                map_dict=dict(zip(encode_order, range(len(encode_order))))
                new_X[col]=new_X[col].map(map_dict)
            return new_X


class CustomScaler(BaseEstimator, TransformerMixin):
    """
        Scale dataframe
    """
    def __init__(self, scaler_type:str = 'standard',
                 scale_col_list=None,
                 exclude_col_list=None):
        self.scaler_type = scaler_type
        self.scale_col_list = scale_col_list
        self.exclude_col_list = exclude_col_list
        self.dummy_encoded_cols=[]
        
    def fit(self, X, y = None):
        all_cols=X.columns.to_list()
        exclude_col_list=[]
        #check scaler_type
        if self.scaler_type=='standard':
            self.scaler=StandardScaler()
        elif self.scaler_type=='minmax':
            self.scaler=MinMaxScaler()
        elif self.scaler_type=='robust':
            self.scaler=RobustScaler()
        else:
            raise Exception("Invalid scaler_type")
        return self
        
    def transform(self, X):
        all_columns=X.columns.to_list()
        #exclude columns that are object or category
        numeric_cols=list(X.select_dtypes(exclude=[object,'category']).columns)
        #print('numeric columns:',numeric_cols)
        #print('exclude columns:',self.exclude_col_list)
        scale_cols=numeric_cols
        #exclude columns that in exclude_col_list
        if (self.exclude_col_list is not None) and len(self.exclude_col_list)>0: 
            for col in self.exclude_col_list:
                #print('check',col)
                if col in numeric_cols:
                    scale_cols.remove(col)
                    #print('remove',col)
        #exclude columns that are dummy encoding(column name contains '_dummy_')
        for col in scale_cols:
            if '_dummy_' in col:
                self.dummy_encoded_cols.append(col)
                scale_cols.remove(col)
                #print('remove',col)
        #print('scale columns:',scale_cols)
        no_scale_cols=[col for col in all_columns if col not in scale_cols]
        #print('no scale columns:',no_scale_cols)
        scaled_X=self.scaler.fit_transform(X[scale_cols])
        df_scaled_X = pd.DataFrame(scaled_X, index=X[scale_cols].index, columns=X[scale_cols].columns)
        result=pd.merge(X[no_scale_cols], df_scaled_X, how="inner",left_index=True, right_index=True)
        return result