import pandas as pd
import os

def csv_to_markdown_pandas(csv_file_path):
    if not os.path.exists(csv_file_path):
        print("文件不存在")
        return

    try:
        # 读取CSV
        df = pd.read_csv(csv_file_path)
        
        # 转换为Markdown
        # index=False 表示不包含行号索引
        markdown_table = df.to_markdown(index=False)
        
        print(markdown_table)
        
        # 如果你想保存到文件：
        with open(csv_file_path.replace(".csv", ".md"), "w", encoding="utf-8") as f:
            f.write(markdown_table)
            
    except Exception as e:
        print(f"转换失败: {e}")

if __name__ == "__main__":
    # 创建示例文件
    # with open("test_pandas.csv", "w", encoding="utf-8") as f:
    #     f.write("ID,Name,Score\n1,Alice,95\n2,Bob,88")

    csv_to_markdown_pandas("results/basic_lm_benchmark_results_warmups2.csv")