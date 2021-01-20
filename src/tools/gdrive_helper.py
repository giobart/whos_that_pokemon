import requests

def download_file_from_google_drive(full_url, destination):
    if 'google' in full_url:
        URL = "https://docs.google.com/uc?export=download"
        id = full_url.split('id=')[-1]
        print('id', id)
        session = requests.Session()

        response = session.get(URL, params = { 'id' : id }, stream = True)
        token = get_confirm_token(response)

        if token:
            params = { 'id' : id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)

        save_response_content(response, destination)
    else:
        r = requests.get(full_url, allow_redirects=True)
        with open(destination, 'wb') as f:
            f.write(r.content)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    print('Downloading is done!')
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)