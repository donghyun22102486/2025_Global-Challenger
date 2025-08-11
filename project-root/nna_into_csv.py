import pandas as pd

# .NNA 파일 읽기 (탭 구분)
df = pd.read_csv("project-root/example-data/Faults.NNA", sep="\t", header=None)

# CSV로 저장
df.to_csv("project-root/example-data/Faults.csv", index=False)
