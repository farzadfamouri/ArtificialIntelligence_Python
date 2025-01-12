import urllib.request

# List of URLs and corresponding filenames
files = [
    ("https://assets.datacamp.com/production/repositories/655/datasets/1304e66b1f9799e1a5eac046ef75cf57bb1dd630/company-stock-movements-2010-2015-incl.csv", "company_stock_movements.csv"),
    ("https://assets.datacamp.com/production/repositories/655/datasets/2a1f3ab7bcc76eef1b8e1eb29afbd54c4ebf86f2/eurovision-2016.csv", "eurovision_2016.csv"),
    ("https://assets.datacamp.com/production/repositories/655/datasets/fee715f8cf2e7aad9308462fea5a26b791eb96c4/fish.csv", "fish.csv"),
    ("https://assets.datacamp.com/production/repositories/655/datasets/effd1557b8146ab6e620a18d50c9ed82df990dce/lcd-digits.csv", "lcd_digits.csv"),
    ("https://assets.datacamp.com/production/repositories/655/datasets/2b27d4c4bdd65801a3b5c09442be3cb0beb9eae0/wine.csv", "wine.csv")
]

# Download each file
for url, filename in files:
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename} from {url}")