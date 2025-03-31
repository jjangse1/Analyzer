import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy import stats
from dtaidistance import dtw
import io
import matplotlib.pyplot as plt

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
            # Use error_bad_lines and warn_bad_lines for handling problematic rows
            df = pd.read_csv(file, parse_dates=['time'], on_bad_lines='warn')
            
            # Remove rows with NaN in step column
            df = df.dropna(subset=['step'])
            
            # Ensure step is integer, converting any float/string to int
            df['step'] = pd.to_numeric(df['step'], errors='coerce').fillna(0).astype(int)
            
            # Remove any entirely empty rows
            df = df.dropna(how='all')
            
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
    
    def create_step_aligned_chart(self, ref_dfs, compare_df, ref_filenames, compare_filename):
        """
        Create charts where data is aligned at the beginning of each step
        Modified to handle steps with different numbers of data points
        """
        charts = {}
        
        for sensor in self.sensors:
            fig = go.Figure()
            
            # Get all steps from reference and compare dataframes
            all_steps = set()
            for ref_df in ref_dfs.values():
                all_steps.update(ref_df['step'].unique())
            all_steps.update(compare_df['step'].unique())
            all_steps = sorted(all_steps)
            
            # For each reference dataframe
            for ref_filename, ref_df in ref_dfs.items():
                # Get unique steps in reference dataframe
                ref_steps = ref_df['step'].unique()
                
                # Compare dataframe steps
                compare_steps = compare_df['step'].unique()
                
                # Plot each step separately to create discontinuities
                for i, step in enumerate(all_steps):
                    # Add reference data if step exists in reference file
                    if step in ref_steps:
                        ref_step_data = ref_df[ref_df['step'] == step]
                        
                        # Create normalized x-axis for alignment, even for single points
                        if len(ref_step_data) > 1:
                            ref_x = np.linspace(i, i+0.9, len(ref_step_data))
                        else:
                            # For single points, still create a visible point
                            ref_x = [i + 0.45]  # Center the point
                        
                        # Add reference data for this step
                        fig.add_trace(go.Scatter(
                            x=ref_x,
                            y=ref_step_data[sensor],
                            mode='lines+markers' if len(ref_step_data) > 1 else 'markers',
                            name=f'Ref {ref_filename} - Step {step}',
                            line=dict(color='purple'),
                            marker=dict(size=6),
                            legendgroup=f'Ref {ref_filename}',
                            showlegend=i==0  # Only show in legend once per file
                        ))
                    
                    # Add compare data if step exists in compare file
                    if step in compare_steps:
                        compare_step_data = compare_df[compare_df['step'] == step]
                        
                        # Create normalized x-axis for alignment, even for single points
                        if len(compare_step_data) > 1:
                            compare_x = np.linspace(i, i+0.9, len(compare_step_data))
                        else:
                            # For single points, still create a visible point
                            compare_x = [i + 0.45]  # Center the point
                        
                        # Add compare data for this step
                        fig.add_trace(go.Scatter(
                            x=compare_x,
                            y=compare_step_data[sensor],
                            mode='lines+markers' if len(compare_step_data) > 1 else 'markers',
                            name=f'Compare {compare_filename} - Step {step}',
                            line=dict(color='red'),
                            marker=dict(size=6),
                            legendgroup=f'Compare {compare_filename}',
                            showlegend=i==0  # Only show in legend once per file
                        ))
                
                # Add vertical lines to indicate step boundaries
                for i in range(1, len(all_steps)):
                    fig.add_shape(
                        type="line",
                        x0=i, x1=i,
                        y0=0, y1=1,
                        yref="paper",
                        line=dict(color="gray", width=1, dash="dash")
                    )
            
            # Update layout
            fig.update_layout(
                title=f'{sensor.replace("_", " ").title()} Step-Aligned Comparison',
                xaxis_title='Process Steps (Aligned)',
                yaxis_title='Sensor Value',
                height=400,
                legend_title="Data Sources",
                # 범례 클릭 시 완전히 숨기도록 설정
                legend=dict(
                    itemclick="toggleothers",  # 클릭 시 다른 모든 트레이스 토글
                    itemdoubleclick="toggle"   # 더블 클릭 시 해당 트레이스만 토글
                )
            )
            
            # Add step labels on x-axis
            if len(all_steps) > 0:
                fig.update_xaxes(
                    tickvals=np.arange(0.5, len(all_steps)),
                    ticktext=[f'Step {step}' for step in all_steps]
                )
            
            # Create unique key for the chart
            unique_key = f"{'_'.join(ref_filenames)}_{compare_filename}_{sensor}_step_aligned_chart"
            charts[sensor] = (fig, unique_key)
        
        return charts

    def create_comprehensive_step_aligned_overlay(self, ref_dfs, compare_dfs, tab_prefix=""):
        """
        Create step-aligned charts for all reference and comparison files together
        using the same format as Individual Comparisons
        """
        comprehensive_charts = {}
        
        for sensor in self.sensors:
            fig = go.Figure()
            
            # Get all steps from all dataframes
            all_steps = set()
            for df in list(ref_dfs.values()) + list(compare_dfs.values()):
                all_steps.update(df['step'].unique())
            all_steps = sorted(all_steps)
            
            # Color palette for different files
            ref_colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(ref_dfs)))
            compare_colors = plt.cm.Reds(np.linspace(0.5, 0.9, len(compare_dfs)))
            
            # Process reference files
            for idx, (ref_filename, ref_df) in enumerate(ref_dfs.items()):
                ref_steps = ref_df['step'].unique()
                ref_color = f'#{int(ref_colors[idx][0]*255):02x}{int(ref_colors[idx][1]*255):02x}{int(ref_colors[idx][2]*255):02x}'
                
                # Plot each step separately to create discontinuities
                for i, step in enumerate(all_steps):
                    if step in ref_steps:
                        ref_step_data = ref_df[ref_df['step'] == step]
                        
                        # Create normalized x-axis for alignment
                        if len(ref_step_data) > 1:
                            ref_x = np.linspace(i, i+0.9, len(ref_step_data))
                        else:
                            ref_x = [i + 0.45]  # Center the point
                        
                        # Add reference data for this step
                        fig.add_trace(go.Scatter(
                            x=ref_x,
                            y=ref_step_data[sensor],
                            mode='lines+markers' if len(ref_step_data) > 1 else 'markers',
                            name=f'Ref {ref_filename} - Step {step}',
                            line=dict(color=ref_color),
                            marker=dict(size=6),
                            legendgroup=f'Ref {ref_filename}',
                            showlegend=i==0  # Only show in legend once per file
                        ))
            
            # Process compare files
            for idx, (compare_filename, compare_df) in enumerate(compare_dfs.items()):
                compare_steps = compare_df['step'].unique()
                compare_color = f'#{int(compare_colors[idx][0]*255):02x}{int(compare_colors[idx][1]*255):02x}{int(compare_colors[idx][2]*255):02x}'
                
                # Plot each step separately
                for i, step in enumerate(all_steps):
                    if step in compare_steps:
                        compare_step_data = compare_df[compare_df['step'] == step]
                        
                        # Create normalized x-axis for alignment
                        if len(compare_step_data) > 1:
                            compare_x = np.linspace(i, i+0.9, len(compare_step_data))
                        else:
                            compare_x = [i + 0.45]  # Center the point
                        
                        # Add compare data for this step
                        fig.add_trace(go.Scatter(
                            x=compare_x,
                            y=compare_step_data[sensor],
                            mode='lines+markers' if len(compare_step_data) > 1 else 'markers',
                            name=f'Compare {compare_filename} - Step {step}',
                            line=dict(color=compare_color),
                            marker=dict(size=6),
                            legendgroup=f'Compare {compare_filename}',
                            showlegend=i==0  # Only show in legend once per file
                        ))
            
            # Add vertical lines to indicate step boundaries
            for i in range(1, len(all_steps)):
                fig.add_shape(
                    type="line",
                    x0=i, x1=i,
                    y0=0, y1=1,
                    yref="paper",
                    line=dict(color="gray", width=1, dash="dash")
                )
            
            # Update layout
            fig.update_layout(
                title=f'{sensor.replace("_", " ").title()} Step-Aligned Comparison (All Files)',
                xaxis_title='Process Steps (Aligned)',
                yaxis_title='Sensor Value',
                height=400,
                legend_title="Data Sources",
                legend=dict(
                    itemclick="toggleothers",
                    itemdoubleclick="toggle"
                )
            )
            
            # Add step labels on x-axis
            if len(all_steps) > 0:
                fig.update_xaxes(
                    tickvals=np.arange(0.5, len(all_steps)),
                    ticktext=[f'Step {step}' for step in all_steps]
                )
            
            # Create unique key for the chart
            unique_key = f"{tab_prefix}comprehensive_overlay_{sensor}"
            comprehensive_charts[sensor] = (fig, unique_key)
        
        return comprehensive_charts

    
    # def create_aggregate_charts(self, ref_dfs, compare_dfs, view_mode='time'):
    #     """
    #     Create aggregate charts for each sensor with all reference and comparison files
    #     """
    #     aggregate_charts = {}
        
    #     for sensor in self.sensors:
    #         # 새로운 Figure 생성
    #         fig = go.Figure()
            
    #         # 참조 파일들 추가
    #         for ref_filename, ref_df in ref_dfs.items():
    #             # 데이터 복사 및 정렬
    #             ref_df_aligned = ref_df.copy()
                
    #             if view_mode == 'step':
    #                 # 스텝 기준 정렬
    #                 min_ref_step = ref_df_aligned['step'].min()
    #                 ref_df_aligned['step'] = ref_df_aligned['step'] - min_ref_step
    #                 x_ref = ref_df_aligned['step']
    #             else:
    #                 x_ref = ref_df_aligned['time']
                
    #             # 참조 데이터 추가 (파란색 계열)
    #             fig.add_trace(go.Scatter(
    #                 x=x_ref, 
    #                 y=ref_df_aligned[sensor], 
    #                 mode='lines', 
    #                 name=f'Ref {ref_filename}',
    #                 line=dict(color='blue', width=1, dash='dot')
    #             ))
            
    #         # 비교 파일들 추가
    #         for compare_filename, compare_df in compare_dfs.items():
    #             # 데이터 복사 및 정렬
    #             compare_df_aligned = compare_df.copy()
                
    #             if view_mode == 'step':
    #                 # 스텝 기준 정렬
    #                 min_compare_step = compare_df_aligned['step'].min()
    #                 compare_df_aligned['step'] = compare_df_aligned['step'] - min_compare_step
    #                 x_compare = compare_df_aligned['step']
    #             else:
    #                 x_compare = compare_df_aligned['time']
                
    #             # 비교 데이터 추가 (빨간색 계열)
    #             fig.add_trace(go.Scatter(
    #                 x=x_compare, 
    #                 y=compare_df_aligned[sensor], 
    #                 mode='lines', 
    #                 name=f'Compare {compare_filename}',
    #                 line=dict(color='red', width=1, dash='dot')
    #             ))
            
    #         # 레이아웃 설정
    #         fig.update_layout(
    #             title=f'Aggregate {sensor.replace("_", " ").title()} Comparison',
    #             xaxis_title='Time' if view_mode == 'time' else 'Step',
    #             yaxis_title='Sensor Value',
    #             height=400,
    #             legend_title_text='Files'
    #         )
            
    #         # 고유 키 생성
    #         unique_key = f"aggregate_{sensor}_{view_mode}_chart"
    #         aggregate_charts[sensor] = (fig, unique_key)
        
    #     return aggregate_charts
    
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
                tab1, tab2, tab3, tab4 ,tab5 = st.tabs([
                "Comparative Charts", 
                "Correlation Analysis", 
                "Generate Report",
                # "Aggregate Charts",
                "Step-Aligned Charts",  # New tab for step-aligned charts
                "Step Detail Anlysis"
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
                        columns = st.columns(2)
                        chart_items = list(comparative_charts.items())
                        for i, (sensor, (fig, unique_key)) in enumerate(chart_items):
                            with columns[i % 2]:
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
                with tab4:
                    st.subheader("Step-Aligned Charts")
                    # Radio button to select chart type
                    chart_type = st.radio(
                        "Chart Type",
                        ["Individual Comparisons", "Comprehensive Overlay"]
                    )

                    if chart_type == "Individual Comparisons":
                        # Create step-aligned charts for each comparison file
                        for compare_file in compare_files:
                            compare_df = dataframes[compare_file]
                            st.subheader(f"Step-Aligned: {', '.join(ref_files)} vs {compare_file}")
                            
                            # Create step-aligned charts
                            step_aligned_charts = self.create_step_aligned_chart(
                                ref_dfs, compare_df, ref_files, compare_file
                            )
                            
                            # Display charts for each sensor
                            columns = st.columns(2)
                            chart_items = list(step_aligned_charts.items())
                            for i, (sensor, (fig, unique_key)) in enumerate(chart_items):
                                with columns[i % 2]:
                                    st.plotly_chart(fig, use_container_width=True, key=unique_key)
                    
                    else: # Comprehensive Overlay
                        # Create comprehensive step-aligned overlay with all files
                        comprehensive_charts = self.create_comprehensive_step_aligned_overlay(
                            ref_dfs, # Reference dataframes
                            {file: dataframes[file] for file in compare_files}, # Compare dataframes
                            tab_prefix="tab4_"  # 고유 접두사 추가
                        )
                        
                        # Display charts for each sensor
                        columns = st.columns(2)
                        chart_items = list(comprehensive_charts.items())
                        for i, (sensor, (fig, unique_key)) in enumerate(chart_items):
                            with columns[i % 2]:
                                st.plotly_chart(fig, use_container_width=True, key=unique_key)

                # tab5 부분 (새로운 디테일 분석 탭)
                with tab5:
                    st.subheader("Step Detail Analysis")
                    
                    # 모든 데이터프레임에서 사용 가능한 스텝 추출
                    all_steps = set()
                    for df in list(ref_dfs.values()) + list(dataframes[file] for file in compare_files):
                        all_steps.update(df['step'].unique())
                    all_steps = sorted(all_steps)
                    
                    # 사용자가 분석할 스텝 선택
                    selected_step = st.selectbox("Select Step to Analyze", all_steps)
                    
                    # 선택된 스텝에 대한 데이터 필터링
                    ref_step_dfs = {name: df[df['step'] == selected_step] for name, df in ref_dfs.items() if selected_step in df['step'].unique()}
                    compare_step_dfs = {name: df[df['step'] == selected_step] for name, df in {file: dataframes[file] for file in compare_files}.items() 
                                    if selected_step in df['step'].unique()}
                    
                    # 차트 타입 선택
                    chart_type = st.radio(
                        "Chart Type",
                        ["Statistical Summary", "Distribution Analysis", "Time Series"]
                    )
                    
                    if chart_type == "Statistical Summary":
                        # 통계 요약 정보 표시
                        st.subheader(f"Statistical Summary for Step {selected_step}")
                        
                        # 센서별 통계 차트 생성
                        columns = st.columns(2)
                        for i, sensor in enumerate(self.sensors):
                            with columns[i % 2]:
                                fig = go.Figure()
                                
                                # 참조 파일 데이터 추가
                                for name, df in ref_step_dfs.items():
                                    if not df.empty and sensor in df.columns:
                                        fig.add_trace(go.Box(
                                            y=df[sensor],
                                            name=f"Ref: {name}",
                                            boxmean=True,
                                            marker_color='blue'
                                        ))
                                
                                # 비교 파일 데이터 추가
                                for name, df in compare_step_dfs.items():
                                    if not df.empty and sensor in df.columns:
                                        fig.add_trace(go.Box(
                                            y=df[sensor],
                                            name=f"Compare: {name}",
                                            boxmean=True,
                                            marker_color='red'
                                        ))
                                
                                fig.update_layout(
                                    title=f"{sensor.replace('_', ' ').title()}",
                                    yaxis_title="Value",
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True, key=f"stat_summary_{sensor}_{selected_step}")
                    
                    elif chart_type == "Distribution Analysis":
                        # 분포 분석 차트 표시
                        st.subheader(f"Distribution Analysis for Step {selected_step}")
                        
                        columns = st.columns(2)
                        for i, sensor in enumerate(self.sensors):
                            with columns[i % 2]:
                                fig = go.Figure()
                                
                                # 참조 파일 데이터 추가
                                for name, df in ref_step_dfs.items():
                                    if not df.empty and sensor in df.columns:
                                        fig.add_trace(go.Histogram(
                                            x=df[sensor],
                                            name=f"Ref: {name}",
                                            opacity=0.7,
                                            marker_color='blue'
                                        ))
                                
                                # 비교 파일 데이터 추가
                                for name, df in compare_step_dfs.items():
                                    if not df.empty and sensor in df.columns:
                                        fig.add_trace(go.Histogram(
                                            x=df[sensor],
                                            name=f"Compare: {name}",
                                            opacity=0.7,
                                            marker_color='red'
                                        ))
                                
                                fig.update_layout(
                                    title=f"{sensor.replace('_', ' ').title()} Distribution",
                                    xaxis_title="Value",
                                    yaxis_title="Count",
                                    barmode='overlay',
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True, key=f"dist_analysis_{sensor}_{selected_step}")
                    
                    else:  # Time Series
                        # 시계열 분석 차트 표시
                        st.subheader(f"Time Series Analysis for Step {selected_step}")
                        
                        columns = st.columns(2)
                        for i, sensor in enumerate(self.sensors):
                            with columns[i % 2]:
                                fig = go.Figure()
                                
                                # 참조 파일 데이터 추가
                                for name, df in ref_step_dfs.items():
                                    if not df.empty and sensor in df.columns:
                                        fig.add_trace(go.Scatter(
                                            x=df.index,
                                            y=df[sensor],
                                            mode='lines+markers',
                                            name=f"Ref: {name}",
                                            line=dict(color='blue')
                                        ))
                                
                                # 비교 파일 데이터 추가
                                for name, df in compare_step_dfs.items():
                                    if not df.empty and sensor in df.columns:
                                        fig.add_trace(go.Scatter(
                                            x=df.index,
                                            y=df[sensor],
                                            mode='lines+markers',
                                            name=f"Compare: {name}",
                                            line=dict(color='red')
                                        ))
                                
                                fig.update_layout(
                                    title=f"{sensor.replace('_', ' ').title()} Time Series",
                                    xaxis_title="Time",
                                    yaxis_title="Value",
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True, key=f"time_series_{sensor}_{selected_step}")
                # with tab5:
                #     st.subheader("Step-Aligned Charts")
                #     # Radio button to select chart type
                #     chart_type = st.radio(
                #         "Chart Type",
                #         ["Individual Comparisons", "Comprehensive Overlay"]
                #     )

                #     if chart_type == "Individual Comparisons":
                #         # Create step-aligned charts for each comparison file
                #         for compare_file in compare_files:
                #             compare_df = dataframes[compare_file]
                #             st.subheader(f"Step-Aligned: {', '.join(ref_files)} vs {compare_file}")
                            
                #             # Create step-aligned charts
                #             step_aligned_charts = self.create_step_aligned_chart(
                #                 ref_dfs, compare_df, ref_files, compare_file
                #             )
                            
                #             # Display charts for each sensor
                #             columns = st.columns(3)
                #             chart_items = list(step_aligned_charts.items())
                #             for i, (sensor, (fig, unique_key)) in enumerate(chart_items):
                #                 with columns[i % 3]:
                #                     st.plotly_chart(fig, use_container_width=True, key=unique_key)
                    
                #     else: # Comprehensive Overlay
                #         # Create comprehensive step-aligned overlay with all files
                #         comprehensive_charts = self.create_comprehensive_step_aligned_overlay(
                #         ref_dfs, # Reference dataframes
                #         {file: dataframes[file] for file in compare_files}, # Compare dataframes
                #         tab_prefix="tab5_"  # 고유 접두사 추가
                #     )
                        
                #         # Display charts for each sensor
                #         columns = st.columns(3)
                #         chart_items = list(comprehensive_charts.items())
                #         for i, (sensor, (fig, unique_key)) in enumerate(chart_items):
                #             with columns[i % 3]:
                #                 st.plotly_chart(fig, use_container_width=True, key=unique_key)

        pass
    
if __name__ == "__main__":
    analyzer = SemiconductorProcessAnalyzer()
    analyzer.main()
