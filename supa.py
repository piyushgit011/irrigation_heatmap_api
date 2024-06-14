from supabase import create_client, Client
import io
import tempfile
# Initialize the Supabase client
url = ""
key = ""
supabase: Client = create_client(url, key)

def upload_image_to_supabase(image_data: io.BytesIO, file_name: str) -> str:
    """
    Uploads image data from memory to Supabase Storage and retrieves the public URL.

    :param bucket: Name of the Supabase Storage bucket
    :param image_data: A BytesIO object containing the image data
    :param file_name: The name of the file as it will be stored in Supabase
    :return: Public URL of the uploaded image
    """
    # Seek to the beginning of the BytesIO stream
    bucket = "images"
    image_data.seek(0)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        # Ensure the pointer is at the start
        image_data.seek(0)
        # Copy data from BytesIO to tmp_file
        tmp_file.write(image_data.read())
        tmp_file_path = tmp_file.name
    # Upload the image to Supabase Storage
    response = supabase.storage.from_(bucket).upload(file_name, tmp_file_path)

    # Check for any errors in the response
    # if response.get('error'):
    #     raise Exception(response['error']['message'])

    # Get the public URL of the uploaded image
    public_url_response = supabase.storage.from_(bucket).get_public_url(file_name)
    # if public_url_response.get('error'):
    #     raise Exception(public_url_response['error']['message'])
    # print(public_url_response)
    return public_url_response


