import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy import stats
from dtaidistance import dtw
import io

class SemiconductorProcessAnalyzer:
    def __init__(self):
        st.set_page_config(page_title="Semiconductor Process Log Analysis", layout="wide")
        self.sensors = ['rf_power', 'esc_inner_temp', 'esc_outer_temp', 'wall_temp', 'rf_ref_power']
    
    def load_data(self, uploaded_files):
        """
        Load multiple CSV files and create a dictionary of dataframes
        """
        dataframes = {}
        for file in uploaded_files:
            filename = file.name
            df = pd.read_csv(file, parse_dates=['time'])
            # Ensure step is integer
            df['step'] = df['step'].astype(int)
            dataframes[filename] = df
        
        return dataframes
    
    def calculate_dtw_score(self, ref_vector, compare_vector):
        """
        Calculate DTW score with normalization to prevent vector errors
        """
        # 표준편차가 0인 경우 체크
        ref_std = np.std(ref_vector)
        compare_std = np.std(compare_vector)
        
        # 표준편차가 0이면 평균만 빼고 나누지 않음
        if ref_std == 0:
            ref_normalized = ref_vector - np.mean(ref_vector)
        else:
            ref_normalized = (ref_vector - np.mean(ref_vector)) / ref_std
        
        if compare_std == 0:
            compare_normalized = compare_vector - np.mean(compare_vector)
        else:
            compare_normalized = (compare_vector - np.mean(compare_vector)) / compare_std
        
        # Pad or truncate to ensure same length
        max_length = max(len(ref_normalized), len(compare_normalized))
        ref_normalized = np.pad(ref_normalized, (0, max_length - len(ref_normalized)), mode='constant')
        compare_normalized = np.pad(compare_normalized, (0, max_length - len(compare_normalized)), mode='constant')
        
        try:
            distance = dtw.distance(ref_normalized, compare_normalized)
            return distance
        except Exception as e:
            st.error(f"DTW Calculation Error: {e}")
            return np.nan
    
    def align_data_by_step(self, ref_df, compare_df):
        """
        Align reference and comparison dataframes by step
        """
        # Find the minimum starting step for both dataframes
        min_ref_step = ref_df['step'].min()
        min_compare_step = compare_df['step'].min()
        
        # Filter and reset step index
        ref_aligned = ref_df[ref_df['step'] >= min_ref_step].copy()
        ref_aligned['step'] = ref_aligned['step'] - min_ref_step
        
        compare_aligned = compare_df[compare_df['step'] >= min_compare_step].copy()
        compare_aligned['step'] = compare_aligned['step'] - min_compare_step
        
        return ref_aligned, compare_aligned
    
    def create_comparative_chart(self, ref_dfs, compare_df, ref_filenames, compare_filename, view_mode='time'):
        """
        Create comparative charts for each sensor with time or step view
        """
        charts = {}
        
        for sensor in self.sensors:
            fig = go.Figure()
            
            # 모든 참조 파일에 대해 차트 추가
            for ref_filename, ref_df in ref_dfs.items():
                # 데이터 정렬 및 복사본 생성
                ref_df_aligned = ref_df.copy()
                compare_df_aligned = compare_df.copy()
                
                if view_mode == 'step':
                    # 스텝 기준 정렬
                    min_ref_step = ref_df_aligned['step'].min()
                    min_compare_step = compare_df_aligned['step'].min()
                    
                    ref_df_aligned['step'] = ref_df_aligned['step'] - min_ref_step
                    compare_df_aligned['step'] = compare_df_aligned['step'] - min_compare_step
                    
                    x_ref = ref_df_aligned['step']
                    x_compare = compare_df_aligned['step']
                else:
                    x_ref = ref_df_aligned['time']
                    x_compare = compare_df_aligned['time']
                
                # 참조 데이터 추가
                fig.add_trace(go.Scatter(
                    x=x_ref, 
                    y=ref_df_aligned[sensor], 
                    mode='lines', 
                    name=f'Reference {ref_filename} {sensor}',
                    line=dict(color='blue')
                ))
            
            # 비교 데이터 추가
            fig.add_trace(go.Scatter(
                x=x_compare, 
                y=compare_df_aligned[sensor], 
                mode='lines', 
                name=f'Compare {compare_filename} {sensor}',
                line=dict(color='red')
            ))
            
            # DTW 점수 계산
            dtw_scores = [
                self.calculate_dtw_score(ref_df[sensor].values, compare_df[sensor].values) 
                for ref_df in ref_dfs.values()
            ]
            avg_dtw_score = np.mean(dtw_scores)
            
            fig.update_layout(
                title=f'{sensor.replace("_", " ").title()} Comparison (Avg DTW Score: {avg_dtw_score:.4f})',
                xaxis_title='Time' if view_mode == 'time' else 'Step',
                yaxis_title='Sensor Value',
                height=400
            )
            
            # 고유 키 생성
            unique_key = f"{'_'.join(ref_filenames)}_{compare_filename}_{sensor}_{view_mode}_chart"
            charts[sensor] = (fig, unique_key)
        
        return charts

    # generate_report 메서드 수정
    def generate_report(self, ref_dfs, compare_df, all_dtw_scores):
        """
        Generate a comprehensive CSV report
        """
        # comparison_df에 name 속성 추가
        if not hasattr(compare_df, 'name'):
            compare_df.name = str(compare_df)  # 파일명 또는 식별자 추가
        
        # 나머지 코드는 이전과 동일
        report_data = {
            'Sensor': list(all_dtw_scores.keys()),
            'DTW Score': list(all_dtw_scores.values()),
        }
        
        # 참조 파일 메트릭스 추가
        for i, (ref_filename, ref_df) in enumerate(ref_dfs.items(), 1):
            report_data[f'Ref{i} Filename'] = [ref_filename] * len(self.sensors)
            report_data[f'Ref{i} Mean'] = [ref_df[sensor].mean() for sensor in self.sensors]
            report_data[f'Ref{i} Std'] = [ref_df[sensor].std() for sensor in self.sensors]
        
        # 비교 파일 메트릭스 추가
        report_data['Compare Filename'] = [compare_df.name] * len(self.sensors)
        report_data['Compare Mean'] = [compare_df[sensor].mean() for sensor in self.sensors]
        report_data['Compare Std'] = [compare_df[sensor].std() for sensor in self.sensors]
        
        report_df = pd.DataFrame(report_data)
        return report_df
    
    def main(self):
        st.title("Semiconductor Process Log Comparative Analysis")
    
        # Sidebar for file upload
        st.sidebar.header("Upload Process Log Files")
        uploaded_files = st.sidebar.file_uploader(
            "Choose CSV files", 
            type=['csv'], 
            accept_multiple_files=True
        )
        
        if uploaded_files and len(uploaded_files) > 1:
            # Load data
            dataframes = self.load_data(uploaded_files)
            
            # File selection for reference and comparison
            st.sidebar.header("Select Files for Comparison")
            
            # Reference file selection with multiselect
            ref_files = st.sidebar.multiselect(
                "Select Reference Files", 
                list(dataframes.keys())
            )
            
            # Advanced file selection options
            selection_mode = st.sidebar.radio(
                "File Selection Mode", 
                ["Individual Selection", "Date Range", "Select All"]
            )
            
            compare_files = []
            
            if selection_mode == "Individual Selection":
                # Checkbox selection for individual files
                compare_files = []
                for file in dataframes.keys():
                    if file not in ref_files:
                        if st.sidebar.checkbox(f"Compare with {file}", key=file):
                            compare_files.append(file)
            
            elif selection_mode == "Date Range":
                # Date range selection for each reference file
                date_ranges = {}
                for ref_file in ref_files:
                    ref_start = dataframes[ref_file]['time'].min()
                    ref_end = dataframes[ref_file]['time'].max()
                    date_ranges[ref_file] = (ref_start, ref_end)
                    st.sidebar.write(f"{ref_file} Date Range: {ref_start} to {ref_end}")
                
                # Date range input (using the earliest start and latest end)
                all_starts = [start for start, _ in date_ranges.values()]
                all_ends = [end for _, end in date_ranges.values()]
                overall_start = min(all_starts)
                overall_end = max(all_ends)
                
                start_date = st.sidebar.date_input(
                    "Start Date", 
                    min_value=overall_start, 
                    max_value=overall_end, 
                    value=overall_start
                )
                end_date = st.sidebar.date_input(
                    "End Date", 
                    min_value=start_date, 
                    max_value=overall_end, 
                    value=overall_end
                )
                
                # Filter files within date range
                compare_files = [
                    file for file in dataframes.keys() 
                    if file not in ref_files and 
                    (dataframes[file]['time'].min().date() <= end_date and 
                        dataframes[file]['time'].max().date() >= start_date)
                ]
                
                st.sidebar.write(f"Selected Files: {len(compare_files)}")
            
            elif selection_mode == "Select All":
                # Select all files except reference
                compare_files = [f for f in dataframes.keys() if f not in ref_files]
            
            if ref_files and compare_files:
                # Prepare reference dataframes
                ref_dfs = {ref_file: dataframes[ref_file] for ref_file in ref_files}
                
                # Tabs for different analyses
                tab1, tab2, tab3 = st.tabs([
                    "Comparative Charts", 
                    "Correlation Analysis", 
                    "Generate Report"
                ])
                
                with tab1:
                    # View mode selection
                    view_mode = st.radio("Select View Mode", ['Step', 'Time'])
                    view_mode = view_mode.lower()
                    
                    for compare_file in compare_files:
                        compare_df = dataframes[compare_file]
                        
                        st.subheader(f"Comparison: {', '.join(ref_files)} vs {compare_file}")
                        
                        # Create comparative charts
                        comparative_charts = self.create_comparative_chart(
                            ref_dfs, compare_df, ref_files, compare_file, view_mode
                        )
                        
                        # Display charts for each sensor
                        columns = st.columns(3)
                        chart_items = list(comparative_charts.items())
                        for i, (sensor, (fig, unique_key)) in enumerate(chart_items):
                            with columns[i % 3]:
                                st.plotly_chart(fig, use_container_width=True, key=unique_key)
                
                with tab2:
                    # Perform correlation analysis for each comparison
                    for compare_file in compare_files:
                        compare_df = dataframes[compare_file]
                        st.subheader(f"Correlation: {', '.join(ref_files)} vs {compare_file}")
                        
                        # Combine dataframes for correlation with multiple references
                        combined_columns = []
                        for ref_file, ref_df in ref_dfs.items():
                            ref_cols = ref_df[self.sensors].add_suffix(f'_ref_{ref_file}')
                            combined_columns.append(ref_cols)
                        
                        combined_columns.append(
                            compare_df[self.sensors].add_suffix('_compare')
                        )
                        
                        combined_df = pd.concat(combined_columns, axis=1)
                        correlation_matrix = combined_df.corr()
                        st.dataframe(correlation_matrix)
                
                with tab3:
                    # Generate comprehensive report
                    for compare_file in compare_files:
                        compare_df = dataframes[compare_file]
                        
                        # Calculate DTW scores
                        all_dtw_scores = {}
                        for sensor in self.sensors:
                            # Average DTW score across multiple reference files
                            dtw_scores = [
                                self.calculate_dtw_score(
                                    ref_df[sensor].values, 
                                    compare_df[sensor].values
                                ) for ref_df in ref_dfs.values()
                            ]
                            all_dtw_scores[sensor] = np.mean(dtw_scores)
                        
                        # Generate report
                        report_df = self.generate_report(
                            ref_dfs, 
                            compare_df, 
                            all_dtw_scores
                        )
                        
                        st.subheader(f"Analysis Report: {', '.join(ref_files)} vs {compare_file}")
                        st.dataframe(report_df)
                        
                        # Download report button
                        csv_buffer = io.StringIO()
                        report_df.to_csv(csv_buffer, index=False)
                        
                        st.download_button(
                            label=f"Download Report for {compare_file}",
                            data=csv_buffer.getvalue(),
                            file_name=f'comparison_report_{"_".join(ref_files)}_vs_{compare_file}.csv',
                            mime='text/csv'
                        )

if __name__ == "__main__":
    analyzer = SemiconductorProcessAnalyzer()
    analyzer.main()