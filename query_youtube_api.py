import pandas as pd
from apiclient.discovery import build

def main(all_vids):
    """query youtube for published on dates of all_vids, save to hdf5"""
    
    DEVELOPER_KEY = ""
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"

    data_frame = pd.DataFrame(data=None, columns=['datetime', 'vid', 'category_id'])

    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
    num_in_batch = 50
    
    for batch_ind in range(len(all_vids)/50):
        batch_start = batch_ind*50
        batch_end = (batch_ind + 1)*50
        #in_vids = 'rwEzKUp19vE, a59gmGkq_pw'
        in_vids = ', '.join(all_vids[batch_start: batch_end])
        results = youtube.videos().list(part="snippet", id=in_vids, maxResults=num_in_batch).execute()

        for result_ind, result in enumerate(results['items']):
        
            vid = str(result['id'])
            if vid not in all_vids[batch_start: batch_end]:
                print('not in all_vids!!!!!!!!!!!!!\n\n\n\n')
                
            category_id = int(result['snippet']['categoryId'])
            published_on_str = result['snippet']['publishedAt']

            data_frame.loc[batch_start+result_ind] = pd.to_datetime(published_on_str), vid, category_id
            #print 'Video', batch_start+result_ind, vid, published_on_str, category_id
            print published_on_str.split('.')[1]

    out_file_name = '../vid_published_datetimes.hdf5'
    data_frame.to_hdf(out_file_name, 'w')
