import boto3
from scripts.data_transformer import transform_data


def lambda_handler(event, context):
    # Get uploaded file info
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = event["Records"][0]["s3"]["object"]["key"]

    # Download file
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, "/tmp/raw_data.csv")

    # Process data
    processed_df = transform_data("/tmp/raw_data.csv")
    processed_df.to_csv("/tmp/processed.csv", index=False)

    # Upload processed data
    s3.upload_file(
        "/tmp/processed.csv",
        "nigeria-flood-model-data-eda",
        f'processed/{key.split("/")[-1]}',
    )

    return {"statusCode": 200, "body": "Data processed successfully!"}
