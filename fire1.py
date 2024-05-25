import firebase_admin
from firebase_admin import credentials, storage

cred = credentials.Certificate("i.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'irrigation-17a77.appspot.com'
})


def upload_to_cloud_from_memory(image_data, cloud_file):
    # Configuring Google Cloud Storage
    bucket = storage.bucket()  # Make sure the 'storage' client is properly configured
    blob = bucket.blob(cloud_file)
    
    # Upload from BytesIO (in-memory file)
    blob.upload_from_file(image_data)
    
    # Make the blob publicly viewable
    blob.make_public()
    
    # Return the public URL
    return blob.public_url




