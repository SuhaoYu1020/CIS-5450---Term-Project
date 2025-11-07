import pandas as pd

path1 = "/Users/liyansong/Desktop/final_table_1.xlsx"
path2 = "/Users/liyansong/Desktop/final_table_2.xlsx"
path3 = "/Users/liyansong/Desktop/final_table_3.xlsx"

df1 = pd.read_excel(path1)
df2 = pd.read_excel(path2)
df3 = pd.read_excel(path3)

merged_df = pd.concat([df1, df2, df3], ignore_index=True)

merged_df = merged_df.dropna(axis=0, how='all').dropna(axis=1, how='all')

merged_df = merged_df.rename(columns=lambda x: x.strip())

merged_df['permno'] = merged_df['permno'].astype('Int64')
merged_df['public_year'] = merged_df['public_year'].astype('Int64')

merged_df['tobinq'] = (
    merged_df.groupby(['permno', 'public_year'])['tobinq']
       .transform(lambda s: s.ffill().bfill())
)

merged_df = merged_df.sort_values(['gvkey', 'public_year', 'public_month'])

merged_df['revenue_growth'] = (
    merged_df.groupby('gvkey')['revenue_growth']
      .transform(lambda s: s.ffill().bfill()))

merged_df = merged_df.drop(columns=["prcc_c", "pstk", "pe_op_dil", "shrout", "total_assets", "total_liabilities", "total_asset_growth"])

merged_df = merged_df.dropna(subset=['tobinq', 'revenue_growth', 'quick_ratio', 'curr_ratio', 'inv_turn', 'at_turn', 'de_ratio','pe_op_basic',
                                     'roa', 'roe', 'npm', 'de_ratio'])


output_path = "/Users/liyansong/Desktop/final_table_merged.xlsx"
merged_df.to_excel(output_path, index=False)
