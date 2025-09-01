import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import skew, kurtosis


dataset_names = sns.get_dataset_names()


def safe_plot(plot_func, *args, **kwargs):
    try:
        return plot_func(*args, **kwargs)
    except Exception as e:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Plot failed:\n{str(e)}", ha="center", va="center")
        ax.axis("off")
        return fig


def missing_values_bar(df):
    fig, ax = plt.subplots()
    null_counts = df.isnull().sum()
    bars = ax.bar(null_counts.index, null_counts.values)
    for bar, value in zip(bars, null_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{value}', ha="center", va="bottom")
    ax.tick_params("x", labelrotation=90)
    ax.set_title("Number of Null Values per Feature")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Null Value Count")
    ax.set_ylim(0, max(null_counts) + 10)
    ax.grid(True)
    plt.tight_layout()
    return fig

def detect_outliers(df):
    outlier_dict = {}
    for feature in df.columns:
        if pd.api.types.is_numeric_dtype(df[feature]):
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (df[feature] < Q1 - 1.5 * IQR) | (df[feature] > Q3 + 1.5 * IQR)
            outlier_dict[feature] = [outlier_mask.sum()]
        else:
            outlier_dict[feature] = [0]
    return pd.DataFrame(outlier_dict).T.rename(columns={0: "Number of Outliers"})

def plot_histogram(df, feature):
    fig, ax = plt.subplots()
    sns.histplot(df, x=feature, kde=True, ax=ax)
    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel(feature)
    plt.tight_layout()
    return fig

def plot_boxplot(df, feature):
    fig, ax = plt.subplots()
    sns.boxplot(y=df[feature], ax=ax)
    ax.set_title(f"Distribution of {feature}")
    ax.set_ylabel(feature)
    plt.tight_layout()
    return fig

def plot_scatterplot(df, feature_1, feature_2, hue_feature=None):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=feature_1, y=feature_2, hue=hue_feature, ax=ax)
    title = f"{feature_2} vs. {feature_1}" + (f" by {hue_feature}" if hue_feature else "")
    ax.set_title(title)
    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    plt.tight_layout()
    return fig

def plot_heatmap(df):
    corr = df.select_dtypes("number").corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    plt.tight_layout()
    return fig

def summary_table(df, feature):
    counts = df[feature].value_counts().values
    freq = df[feature].value_counts(normalize=True).values
    subcat = df[feature].value_counts().index
    return pd.DataFrame({
        "Sub-category": subcat,
        "Count": counts,
        "Relative Frequency": freq
    }).set_index("Sub-category")

def plot_bar(df, feature):
    fig, ax = plt.subplots()
    counts = df[feature].value_counts()
    bars = ax.bar(counts.index, counts.values)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{value}', ha="center", va="bottom")
    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig


st.set_page_config(page_title="Automated Seaborn Dataset EDA Tool", layout="wide")
st.title("Automated Seaborn Dataset EDA Tool")



tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Home", "Numerical Univariate", "Numerical Multivariate", "Categorical Univariate", "Categorical Multivariate"
])


with tab1:
    dataset = st.selectbox("Please choose a dataset:", dataset_names, key="dataset_home")
    test_df = sns.load_dataset(dataset)
    st.write(f"You have selected the `{dataset}` dataset")
    st.dataframe(test_df)

    null_value_df = pd.DataFrame(test_df.isnull().sum()).rename(columns={0: "Null Value Count"})
    Dtype_Frame = pd.DataFrame({feature: [test_df[feature].dtype.name] for feature in test_df.columns}).T.rename(columns={0: "Data Type"})
    Dtype_Frame["Type"] = Dtype_Frame["Data Type"].apply(lambda x: "Numerical" if x in ["float64", "int64"] else "Categorical")

    col_null, col_dtype = st.columns(2)
    with col_null:
        st.dataframe(null_value_df)
    with col_dtype:
        st.dataframe(Dtype_Frame)

    st.pyplot(safe_plot(missing_values_bar, test_df))
    st.session_state["dataset"] = test_df


with tab2:
    numerical_features = list(st.session_state["dataset"].select_dtypes(include="number").columns)
    feature_num = st.selectbox("Select numerical feature (Univariate):", numerical_features, key="num_uni_feature")
    
    num_test_df = st.session_state["dataset"][numerical_features]
    skewness = [skew(num_test_df[feature].dropna()) for feature in numerical_features]
    kurt_vals = [kurtosis(num_test_df[feature].dropna()) for feature in numerical_features]

    describe_frame = num_test_df.describe()
    describe_frame.loc["skewness"] = pd.Series(skewness, index=describe_frame.columns)
    describe_frame.loc["kurtosis"] = pd.Series(kurt_vals, index=describe_frame.columns)

    col_describe, col_outlier = st.columns(2)
    with col_describe:
        st.dataframe(describe_frame)
    with col_outlier:
        st.dataframe(detect_outliers(num_test_df))

    col_hist, col_box = st.columns(2)
    with col_hist:
        st.pyplot(safe_plot(plot_histogram, num_test_df, feature_num))
    with col_box:
        st.pyplot(safe_plot(plot_boxplot, num_test_df, feature_num))


with tab3:
    numerical_features = list(st.session_state["dataset"].select_dtypes(include="number").columns)
    num_test_df = st.session_state["dataset"][numerical_features]

    feature1 = st.selectbox("Select X-axis numerical feature:", numerical_features, key="num_multi_x")
    feature2 = st.selectbox("Select Y-axis numerical feature:", numerical_features, key="num_multi_y")

    st.pyplot(safe_plot(plot_scatterplot, num_test_df, feature1, feature2))
    st.pyplot(safe_plot(plot_heatmap, num_test_df))


with tab4:
    categorical_features = list(st.session_state["dataset"].select_dtypes(exclude="number").columns)
    feature_cat = st.selectbox("Select categorical feature (Univariate):", categorical_features, key="cat_uni_feature")
    st.dataframe(summary_table(st.session_state["dataset"], feature_cat))
    st.pyplot(safe_plot(plot_bar, st.session_state["dataset"], feature_cat))


with tab5:
    categorical_features = list(st.session_state["dataset"].select_dtypes(exclude="number").columns)
    feature_cat_1 = st.selectbox("Select first categorical feature:", categorical_features, key="cat_multi_1")
    feature_cat_2 = st.selectbox("Select second categorical feature:", categorical_features, key="cat_multi_2")

    st.dataframe(pd.crosstab(
        st.session_state["dataset"][feature_cat_1],
        st.session_state["dataset"][feature_cat_2]
    ))
