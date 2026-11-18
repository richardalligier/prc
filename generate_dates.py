from datetime import date, timedelta

start = date(2022, 1, 1)
end = date(2022, 12, 31)

current = start
while current <= end:
    print(f"{current}.parquet",end=" ")
    current += timedelta(days=1)
    