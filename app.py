import streamlit as st
from helper import *
def main():
    
    st.title("Auto feature selector tools")
    
    # file uploader
    uploaded_file = st.file_uploader("choose a CSV file",type="csv")
    
    
    if uploaded_file is not None:
        analyze_csv_file(uploaded_file)
        df = st.session_state.get('df')
        
        if df is not None:
            if st.button("Transform object Columns"):
                df, label_encoders = transform_obejct_columns(df)
                st.session_state['df']=df
                st.success("object columns transformed successfully") 
                st.write("### Transformed Data")
                st.write(df)     
            select_target_columns(df)     
            
            feature_selection_options() 
            
            if st.session_state.get('feature_selection_method') == "Correlation Coefficient":
                correlation_coefficient_selection()
                
            if st.session_state.get('feature_selection_method') == "Chi-square test":
                chi_square_selection()
                
            if st.session_state.get('feature_selection_method') == "ANOVA":
                anova_selection()
            
            if st.session_state.get('feature_selection_method') == "Mutual information":
                mutual_information_selection()
                
            if st.session_state.get('feature_selection_method') == "Variance threshold":
                variance_threshold_selection()
    
if __name__ == "__main__":
    main()