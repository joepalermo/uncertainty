import boto3
s3 = boto3.resource('s3')

bucket_name = 'uncertainty'

# download cleaned bulldozer data
s3.meta.client.download_file(bucket_name, 'bulldozer/train.csv', 'data/bulldozer/train.csv')
s3.meta.client.download_file(bucket_name, 'bulldozer/test.csv', 'data/bulldozer/test.csv')
s3.meta.client.download_file(bucket_name, 'bulldozer/categorical_sizes.json', 'data/bulldozer/categorical_sizes.json')
