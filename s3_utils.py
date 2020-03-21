import os
from io import BytesIO

import boto3
from botocore.exceptions import ClientError

S3_DISEASE_DATA_OBJ_NAME = "full_and_latest_disease_data_dict_pkl"

_S3_ACCESS_KEY = os.environ.get("AWSAccessKeyId", "").replace("\r", "")
_S3_SECRET_KEY = os.environ.get("AWSSecretKey","").replace("\r", "")
_S3_BUCKET_NAME = "coronavirus-calculator-data"

DATESTRING_FORMAT_READABLE = "%A %d %B %Y, %H:%M %Z"  # 'Sunday 30 November 2014'


def _configure_client():
    # Upload the file
    s3_client = boto3.client(
        "s3", aws_access_key_id=_S3_ACCESS_KEY, aws_secret_access_key=_S3_SECRET_KEY
    )
    return s3_client


def upload_file(data: bytes, object_name: str):
    """
    Upload a file to an S3 bucket
    :param data: Bytes to upload.
    :param object_name: S3 object name.
    :return: True if file was uploaded, else False
    """
    buf = BytesIO(data)
    s3_client = _configure_client()
    try:
        response = s3_client.put_object(
            Body=buf, Bucket=_S3_BUCKET_NAME, Key=object_name
        )
    except ClientError as e:
        print(e)
        return False
    return True


def download_file_from_s3(object_name: str):
    """
    Download a file from S3 bucket.
    :param object_name: Name of object to download.
    :return: Object bytes and date last modified.
    """
    s3_client = _configure_client()
    try:
        download = s3_client.get_object(Key=object_name, Bucket=_S3_BUCKET_NAME)
    except ClientError:
        return None
    content = download["Body"].read()
    last_modified = download["LastModified"].strftime(DATESTRING_FORMAT_READABLE)
    return content, last_modified
