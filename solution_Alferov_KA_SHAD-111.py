import pandas as pd
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data_train')
df = pd.read_parquet(data_path)
if 'BrandinDelivery' in df.columns:
    filtered_df = df[(df['CategoryNameDelivery'].notna()) & (df['BrandinDelivery'] == 1)]
else:
    filtered_df = df[df['CategoryNameDelivery'].notna()]
group_cols = ['SubjectID', 'researchdate', 'CategoryNameDelivery', 'Brand']
if 'BrandID' in filtered_df.columns:
    group_cols.append('BrandID')
grouped = filtered_df.groupby(group_cols, as_index=False).size()
grouped.rename(columns={'size': 'count_rows'}, inplace=True)

if 'Weight' in filtered_df.columns:
    weights = filtered_df[['SubjectID', 'researchdate', 'Weight']].drop_duplicates()
    grouped = grouped.merge(weights, on=['SubjectID', 'researchdate'], how='left')
    grouped['daily_ots'] = grouped['count_rows'] * grouped['Weight']
else:
    grouped['daily_ots'] = grouped['count_rows'] 

thresholds = grouped.groupby(['CategoryNameDelivery', 'Brand'])['daily_ots'].quantile(0.99).reset_index()
thresholds.rename(columns={'daily_ots': 'threshold'}, inplace=True)
global_min_threshold = grouped['daily_ots'].quantile(0.95)
merged = grouped.merge(thresholds, on=['CategoryNameDelivery', 'Brand'], how='left')
anomalies = merged[(merged['daily_ots'] > merged['threshold']) & (merged['daily_ots'] > global_min_threshold)]

output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

anomalies = anomalies.copy()
anomalies['score'] = anomalies['daily_ots']
anomalies['reason'] = 'Превышен 99-й перц. (' + anomalies['threshold'].round(2).astype(str) + ') И глоб. минимум (' + str(round(global_min_threshold, 2)) + ')'
if 'BrandID' not in anomalies.columns:
    anomalies['BrandID'] = ''
reasons_path = os.path.join(output_dir, 'anomaly_reasons.csv')
anomalies.to_csv(reasons_path, index=False, encoding='utf-8-sig')

to_delete = anomalies[['SubjectID', 'researchdate']].drop_duplicates()
anomalies_path = os.path.join(output_dir, 'anomalies.csv')
to_delete.to_csv(anomalies_path, index=False, encoding='utf-8-sig')

print(f"Глобальный минимальный порог: {global_min_threshold}")
print(f"Сохранено {len(to_delete)} уникальных дней-респондентов для удаления.")

plots_dir = os.path.join(output_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

clean_data = grouped.merge(to_delete, on=['SubjectID', 'researchdate'], how='left', indicator=True)
clean_data = clean_data[clean_data['_merge'] == 'left_only'].drop(columns=['_merge'])



ots_before = grouped.groupby('researchdate')['daily_ots'].sum()
ots_after = clean_data.groupby('researchdate')['daily_ots'].sum()
plt.figure(figsize=(12, 6))
plt.plot(ots_before.index.astype(str), ots_before.values, label='OTS До', color='red')
plt.plot(ots_after.index.astype(str), ots_after.values, label='OTS После', color='green')
plt.title('Изменение общего OTS по дням')
plt.xlabel('Дата')
plt.ylabel('Суммарный OTS')
dates_1 = ots_before.index.astype(str)
plt.xticks(ticks=range(0, len(dates_1), 5), labels=dates_1[::5], rotation=45, fontsize=8)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'total_ots_before_after.png'))
plt.close()

cat_before = grouped.groupby('CategoryNameDelivery')['daily_ots'].sum()
cat_after = clean_data.groupby('CategoryNameDelivery')['daily_ots'].sum()
cat_change = ((cat_after - cat_before) / cat_before * 100).fillna(0)
top_cat_change = cat_change.sort_values().head(20)
plt.figure(figsize=(12, 6))
top_cat_change.plot(kind='bar', color='steelblue')
plt.title('Топ-20 категорий с наибольшим снижением OTS (%)')
plt.xlabel('Категория')
plt.ylabel('Изменение (%)')
plt.axhline(0, color='black', linewidth=0.8)
plt.xticks(rotation=45, fontsize=8) 
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'category_ots_change.png'))
plt.close()


anomalies_per_day = to_delete.groupby('researchdate').size()
plt.figure(figsize=(12, 6))
anomalies_per_day.plot(kind='bar', color='darkorange')
plt.title('Количество аномальных респондентов по дням')
plt.xlabel('Дата')
plt.ylabel('Количество')
dates_3 = anomalies_per_day.index.astype(str)
plt.xticks(ticks=range(0, len(dates_3), 5), labels=dates_3[::5], rotation=45, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'daily_anomaly_count.png'))
plt.close()

print("Графики сохранены в папку output/plots/")

def show_anomalous_queries(subject_id, date_val, raw_df):
    """Возвращает таблицу поисковых запросов для выбранного респондента и дня."""
    user_queries = raw_df[(raw_df['SubjectID'] == subject_id) & (raw_df['researchdate'] == date_val)]
    columns_to_show = ['SubjectID', 'researchdate', 'CategoryNameDelivery', 'Brand']
    if 'QueryText' in user_queries.columns:
        columns_to_show.append('QueryText')
        
    return user_queries[columns_to_show]
if not to_delete.empty:
    sample_id = to_delete.iloc[0]['SubjectID']
    sample_date = to_delete.iloc[0]['researchdate']
    # print(f"\n Поисковые запросы аномального респондента {sample_id} за {sample_date}")
    # print(show_anomalous_queries(sample_id, sample_date, df))

def plot_before_after_distribution(raw_df, delete_df, column_name, title):
    """Строит столбчатую диаграмму до/после по любому категориальному столбцу."""
    if column_name not in raw_df.columns:
        print(f"Столбец {column_name} отсутствует в данных.")
        return
    temp_df = raw_df.copy()
    temp_df['ots'] = temp_df['Weight'] if 'Weight' in temp_df.columns else 1
    ots_before = temp_df.groupby(column_name)['ots'].sum()
    clean_df = temp_df.merge(delete_df, on=['SubjectID', 'researchdate'], how='left', indicator=True)
    clean_df = clean_df[clean_df['_merge'] == 'left_only']
    ots_after = clean_df.groupby(column_name)['ots'].sum()
    df_plot = pd.DataFrame({'OTS До': ots_before, 'OTS После': ots_after}).fillna(0)
    df_plot.plot(kind='bar', figsize=(10, 6), color=['red', 'green'])
    plt.title(title)
    plt.ylabel('Суммарный OTS')
    plt.xlabel(column_name)
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_brand_ots_change(raw_df, delete_df, brand_name):
    """Строит линейный график изменения OTS по дням для выбранного бренда."""
    if 'Brand' not in raw_df.columns:
        return
    temp_df = raw_df[raw_df['Brand'] == brand_name].copy()
    if temp_df.empty:
        print(f"Бренд {brand_name} не найден.")
        return
    temp_df['ots'] = temp_df['Weight'] if 'Weight' in temp_df.columns else 1
    ots_before = temp_df.groupby('researchdate')['ots'].sum()
    clean_df = temp_df.merge(delete_df, on=['SubjectID', 'researchdate'], how='left', indicator=True)
    clean_df = clean_df[clean_df['_merge'] == 'left_only']
    ots_after = clean_df.groupby('researchdate')['ots'].sum()
    plt.figure(figsize=(10, 6))
    dates = ots_before.index.astype(str)
    plt.plot(dates, ots_before.values, label='OTS До', color='red')
    plt.plot(ots_after.index.astype(str), ots_after.values, label='OTS После', color='green')
    plt.title(f'Изменение OTS по дням для бренда: {brand_name}')
    plt.xlabel('Дата')
    plt.ylabel('OTS')
    plt.xticks(ticks=range(0, len(dates), 5), labels=dates[::5], rotation=45, fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.show()

# show_anomalous_queries(sample_id, sample_date, df)
# plot_before_after_distribution(df, to_delete, 'Пол мужчины', 'OTS до и после по полу')
# plot_before_after_distribution(df, to_delete, 'Platform', 'OTS до и после по платформе')
# plot_brand_ots_change(df, to_delete, 'oukitel')