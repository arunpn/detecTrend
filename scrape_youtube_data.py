from lxml import html
import requests, time
import pandas as pd

def main(all_vids):
    """scrape youtube for published on dates of all_vids, save to hdf5"""
    
    data_frame = pd.DataFrame(data=None, columns=['datetime', 'vid'])
    for vid_ind, vid in enumerate(all_vids):
        page_url = 'https://www.youtube.com/watch?v=' + vid
        page = requests.get(page_url)
        tree = html.fromstring(page.content)
        watch_time_text = tree.xpath('//strong[@class="watch-time-text"]')
        
        if len(watch_time_text) > 0:
            published_on_str = watch_time_text[0].text.split('on ')[1]
        else:
            published_on_str = ''
            
        data_frame.loc[vid_ind] = pd.to_datetime(published_on_str), vid
        print 'Page', vid_ind, vid, published_on_str

    out_file_name = '../vid_published_dates.hdf5'
    data_frame.to_hdf(out_file_name, 'w')
    
    return
