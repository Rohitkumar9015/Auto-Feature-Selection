import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from fancyimpute import IterativeImputer

from sklearn.preprocessing import LabelEncoder

from io import StringIO
from sklearn.feature_selection import SelectKBest,chi2 ,f_classif , mutual_info_classif
from sklearn.feature_selection import VarianceThreshold


def analyze_csv_file(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = st.session_state.get('df',df) # permanent file to not data repeat multiple time
        st.write(df)
        
        if 'columns_removed' not in st.session_state:
            st.write(f"**Columns:**{list(df.columns)}") # showing the columns in list format
            selected_columns = st.multiselect("Select columns to remove",options=df.columns)# provied the columns to select and save the select columns in selected_columns
            if st.button("Remove Selected columns"):
                if selected_columns:
                    df = df.drop(columns=selected_columns)
                    st.session_state['df']=df
                    st.session_state['columns_removed']=True
                    st.success(f"columns removed:{','.join(selected_columns)}")
                    st.write("### Dateframe After removing selected columns")
                    st.write(df)
        missing_values = df.isnull().sum()
        has_missing_values = missing_values.sum()>0
        
        duplicate_count = df.duplicated().sum()
        has_duplicates = duplicate_count>0
        
        if has_missing_values or has_duplicates:
            st.warning("There are missing/duplicate values with the uploaded csv file")
            
            if has_missing_values:
                st.write("### Missing Values")
                st.write(missing_values[missing_values>0])
                
                # optionfor handling missing values
                
                for column in missing_values[missing_values>0].index:
                    st.wrte(f"### Columns:{column}") 
                    
                    # option to remove missing values
                    
                    if st.button(f"Remove rows with missing values in '{column}'"):
                        df = df.dropna(subset=[column])
                        st.session_state['df']=df
                        st.success(f"Rows with missing values in '{column}' remove successfully")
                    # fill with mean
                    if st.button(f"fill missing values in '{column}' with mean"):
                        imputer = SimpleImputer(strategy='mean')
                        df[column]=imputer.fit_transform(df[[column]])
                        st.session_state['df']=df
                        st.success(f"Missing values in '{column}' filled with mean")
                    # fill with median
                    if st.button(f"fill missing values in '{column}' with median"):
                        imputer = SimpleImputer(strategy='median')
                        df[column]=imputer.fit_transform(df[[column]])
                        st.session_state['df']=df
                        st.success(f"Missing values in '{column}' filled with median")
                    # filll with mode
                    if st.button(f" fill missing values in '{column}' with mode"):
                        mode_value = df[column].mode()[0]
                        df[column].fillna(mode_value,inplace=True)
                        st.session_state['df']=df
                        st.success(f"Missing values in '{column}' filled with mode")
                        
                    # fill with custom value
                    custom_value = st.text_input(f"custom value to fill missing values in '{column}'")
                    if st.button(f"Fill missing values in '{column}' with custom value"):
                        if custom_value:
                            df[column].fillna(custom_value,inplace=True)
                            st.session_state['df']=df
                            st.success(f"Missing values in '{column}' filled with custom values")
                        else:
                            st.warning("Please provied a custom values")
                    # iterative imputation(MICE)
                    if st.button(f"apply iterative imputation for'{column}'"):
                        #ensure iterativeImputter is applied to all columns with the missing values
                        imputer=IterativeImputer()
                        df_imputer=pd.DataFrame(imputer.fit_transform(df),columns=df.columns)
                        st.session_state['df']=df_imputer
                        st.success(f"Iterative imputation applied successfully")
                        st.write("### Dataframe with iterative imputation applied")
                        st.write(df_imputer)
        
            if has_duplicates:
                st.write("### Duplicate values")
                st.write(f"**Number of duplicate vlaues:**{duplicate_count}")    
                
                if st.button("Remove Duplicate vlaues"):
                    df = df.drop_duplicates()
                    st.session_state['df']=df
                    st.success("Duplicate rows removed successfully")  
                    duplicate_count=df.duplicated.sum()
                    st.write(duplicate_count)
        show_details = st.checkbox("show details")
        if show_details:
            st.write("### Basic information")
            st.write(f"**Number of columns:**{df.shape[1]}")
            st.write(f"**Columns names:** {list(df.columns)}")
            st.write("### columns data types")
            st.write(df.dtypes)
            st.write("### Missing values")
            st.write(missing_values[missing_values>0])
            st.write("### Duplicate values")
            st.write(f"** Number of dulpicate rows :** {duplicate_count}")
            
def transform_obejct_columns(df):
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column]=le
    return df, label_encoders
def select_target_columns(df):
    if df is not None:
        target_column =st.selectbox("Select target columns",[""]+list(df.columns),index=0)
        if target_column !="":
            X=df.drop(columns=[target_column])
            Y=df[target_column]
            st.session_state['X']=X
            st.session_state['Y']=Y
            st.success(f"target column '{target_column}' selected successfully")
            st.write("### Input Features (x)")
            st.write(X)
            st.write("### Target Feature (Y)")
            st.write(Y)
            
def feature_selection_options():
    feature_selection_method = st.selectbox(
        "Select Feature selection method",
        ["None","Correlation Coefficient","Chi-square test","ANOVA","Mutual information","Variance threshold"],
        index=0
    )
    st.session_state['feature_selection_method']=feature_selection_method
    
    if feature_selection_method != "None":
        st.success(f"'{feature_selection_method}' selected for feature selection")
        
    else:
        st.info("No feature selection method selected.")
        
        
def correlation_coefficient_selection():
    if 'df' in st.session_state and 'Y' in st.session_state:
        df=st.session_state['df']
        Y = st.session_state['Y']
        
        threshold = st.slider("Select Correlation Cofficient threshold",0.0,1.0,0.5)
        
        if st.button("Apply Correlation Cofficient"):
            corr_matrix = df.corrwith(Y)
            
            slected_features = corr_matrix[abs(corr_matrix)>=threshold].index.tolist()
            
            selected_features = [feature for feature in slected_features if feature !=Y.name]
            
            st.write(f"### Selected feature (Correlation Cofficient >={threshold})")
            st.write(selected_features)
            
            if slected_features:
                st.session_state['selected_feature'] = selected_features
                st.success("features selected based o corrrelation cofficient")
            else:
                st.warning("No feature selected based on the given threshold")
                
                
        save_selected_feature()
                
def save_selected_feature():
    if 'selected_feature' in st.session_state and 'Y' in st.session_state:
        df = st.session_state['df']
        selected_feature = st.session_state['selected_feature']
        Y = st.session_state['Y']
        
        new_df = pd.concat([df[selected_feature],Y],axis=1)
        
        #show the new dataframe to the user
        st.write("###New dataframe with selected feature with target columns")
        st.write(new_df)
        
        csv = new_df.to_csv(index=False)
        buffer = StringIO(csv)
        
        #proovide download option 
        st.download_button(
            label="Download csv",
            data=buffer.getvalue(),
            file_name='selected_feature_and_target.csv',
            mime='text/csv'
        )
        
def chi_square_selection():
    if 'X' in st.session_state and 'Y' in st.session_state:
        X = st.session_state['X']
        Y = st.session_state['Y']
        
        #ensure target is categorical for chi square test
        if Y.dtype !='int64':
            st.warning("chi_square thest requires the target columns to be encoded as integers")
            
            return
        # slider for number of top feature to select
        k=st.slider("select the number f top feature to keep ",1,X.shape[1],X.shape[1])
        
        #apply selectkbest with chi2
        selector = SelectKBest(score_func=chi2,k=k)
        X_new = selector.fit_transform(X,Y)
        
        #get selected feature indices
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        #display selected feature
        st.write(f"### Selected Features :")
        st.write(selected_features)
        
        #update session state
        st.session_state['selected_features']=selected_features
        st.success("features selected based on chi square test.")
        
        save_selected_feature()
        
def anova_selection():
    if 'X' in st.session_state and 'Y' in st.session_state:
        X = st.session_state['X']
        Y = st.session_state['Y']
        
        # slider for number of top feature to select
        k=st.slider("select the number f top feature to keep ",1,X.shape[1],X.shape[1])
        
        #apply selectkbest with f_classif (anova)
        selector = SelectKBest(score_func=f_classif ,k=k)
        X_new = selector.fit_transform(X,Y)
        
        #get selected feature indices
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        #display selected feature
        st.write(f"### Selected Features :")
        st.write(selected_features)
        
        #update session state
        st.session_state['selected_features']=selected_features
        st.success("features selected based on anova.")
        
        save_selected_feature()
        
def mutual_information_selection():
    if 'X' in st.session_state and 'Y' in st.session_state:
        X = st.session_state['X']
        Y = st.session_state['Y']
        
        # slider for number of top feature to select
        k=st.slider("select the number f top feature to keep ",1,X.shape[1],X.shape[1])
        
        #apply selectkbest with mutual information 
        selector = SelectKBest(score_func=mutual_info_classif ,k=k)
        X_new = selector.fit_transform(X,Y)
        
        #get selected feature indices
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        #display selected feature
        st.write(f"### Selected Features :")
        st.write(selected_features)
        
        #update session state
        st.session_state['selected_features']=selected_features
        st.success("features selected based on Mutual information.")
        
        save_selected_feature()
        
        
def variance_threshold_selection():
    if 'X' in st.session_state and 'Y' in st.session_state:
        X = st.session_state['X']
        
        # slider for number of top feature to select
        threshold =st.slider("select the variance threshold ",0.0,1.0,0.5)
        
        #apply selectkbest with mutual information 
        selector = VarianceThreshold(threshold=threshold)
        X_new = selector.fit_transform(X)
        
        #get selected feature indices
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        #display selected feature
        st.write(f"### Selected Features :")
        st.write(selected_features)
        
        #update session state
        st.session_state['selected_features']=selected_features
        st.success("features selected based on variance threshold.")
        
        save_selected_feature()