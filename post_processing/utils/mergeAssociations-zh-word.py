import csv
import os
from typing import List, Dict
from statistics import mean
import pandas as pd
from pathlib import Path


class associationMerger:
    def __init__(self, raw_data_divided_path, merged_association_path, lower_threshold: int, upper_threshold: int,
                 annotation_path: Path):
        self.in_data_path = raw_data_divided_path
        self.out_data_path = merged_association_path
        self.out_data_merged_name = raw_data_divided_path.stem+'_merged.csv'
        self.out_data_merged_denoise_name = raw_data_divided_path.stem + '_clean.csv'
        self.threshold_lower = lower_threshold
        self.threshold_upper = upper_threshold
        self.start_sent = (0, 1, 2, 3)
        self.annotation_df = pd.read_csv(annotation_path, dtype={
            'stim_id': str, 'page_id': str, 'item_id': str, 'word_nr': str
        })
        self.mouse_data = self.__read_file()
        self.associations = self.merge_associations()

    # Check if the directory for the files of merged associations exists, if not, make one
    def __make_directory_for_merged_associations(self) -> None:
        if not os.path.exists(self.out_data_path):
            os.mkdir(self.out_data_path)

    def __read_file(self) -> List[Dict]:
        """
        Read in the raw mouse tracking data for one participant
        columns in raw mouse tracking file: submission_id, Index, ItemId, SubjectId, Word, experiment_duration,
        experiment_end_time	experiment_start_time, mousePositionX, mousePositionY, response, responseTime,
        wordPositionBottom, wordPositionLeft, wordPositionRight, wordPositionTop
        """
        with open(self.in_data_path, 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            mouse_data = []
            for row in csvreader:
                entry = {
                    'sbm_id': str(row['submission_id']),
                    'stim_id': str(row['Stimulus']),
                    'page_id': str(row['PageId']),
                    'item_id': str(row['ItemId']),
                    'word_nr': str(row['Index']),
                    'word': str(row['Word']),
                    't': int(row['responseTime']),
                    'x': int(row['mousePositionX']),
                    'y': int(row['mousePositionY']),
                    'trial_id': int(row['trial_id']),
                    'difficulty': str(row['difficulty']),
                    'response': str(row['response']),
                    'rev': str(row['wordRevealPart']),
                }

                # Match with annotation dataframe
                match = self.annotation_df[
                    (self.annotation_df['stim_id'] == entry['stim_id']) &
                    (self.annotation_df['page_id'] == entry['page_id']) &
                    (self.annotation_df['item_id'] == entry['item_id']) &
                    (self.annotation_df['word_nr'] == entry['word_nr'])
                    ]

                if not match.empty:
                    matched_row = match.iloc[0]
                    entry['page_token_id'] = matched_row['page_token_id']
                    entry['para_token_id'] = matched_row['para_token_id']
                    entry['stim_token_id'] = matched_row['stim_token_id']
                    entry['exp_token_id'] = matched_row['exp_token_id']
                    entry['tokens'] = matched_row['tokens']
                else:
                    entry['page_token_id'] = None
                    entry['para_token_id'] = None
                    entry['stim_token_id'] = None
                    entry['exp_token_id'] = None
                    entry['tokens'] = None

                # Clean up & rename keys
                del entry['word']
                del entry['word_nr']

                entry['page_word_id'] = entry.pop('page_token_id')
                entry['para_word_id'] = entry.pop('para_token_id')
                entry['stim_word_id'] = entry.pop('stim_token_id')
                entry['exp_word_id'] = entry.pop('exp_token_id')
                entry['word'] = entry.pop('tokens')

                mouse_data.append(entry)
        # print(mouse_data[0:10])
        return mouse_data

    def merge_associations(self) -> List[List[Dict]]:
        """ merge adjacent data points if they are about the same words to get associations"""
        associations = []
        associations_for_one_item = []
        x_coordinates = []
        y_coordinates = []
        for i in range(len(self.mouse_data)-1):
            if self.mouse_data[i+1]['item_id'] == self.mouse_data[i]['item_id']:
                if self.mouse_data[i+1]['exp_word_id'] == self.mouse_data[i]['exp_word_id']:
                    self.mouse_data[i+1]['t'] = self.mouse_data[i]['t']
                    x_coordinates.append(self.mouse_data[i]['x'])
                    y_coordinates.append(self.mouse_data[i]['y'])

                else:
                    fixed_time = self.mouse_data[i + 1]['t'] - self.mouse_data[i]['t']
                    x_coordinates.append(self.mouse_data[i]['x'])
                    y_coordinates.append(self.mouse_data[i]['y'])
                    merged_association_on_word = {
                        'sbm_id': self.mouse_data[i]['sbm_id'],
                        'stim_id': self.mouse_data[i]['stim_id'], 'page_id': self.mouse_data[i]['page_id'],
                        'item_id': self.mouse_data[i]['item_id'], 'page_word_id': self.mouse_data[i]['page_word_id'],
                        'para_word_id': self.mouse_data[i]['para_word_id'], 'stim_word_id': self.mouse_data[i]['stim_word_id'],
                        'exp_word_id': self.mouse_data[i]['exp_word_id'], 'word': self.mouse_data[i]['word'],
                        'duration': fixed_time, 'start_t': self.mouse_data[i]['t'], 'end_t': self.mouse_data[i+1]['t'],
                        'x_mean': round(mean(x_coordinates), 2), 'y_mean': round(mean(y_coordinates), 2),
                        'response': self.mouse_data[i]['response'], 'rev': self.mouse_data[i]['rev'],
                        'trial_id': self.mouse_data[i]['trial_id'], 'difficulty': self.mouse_data[i]['difficulty']
                    }
                    associations_for_one_item.append(merged_association_on_word)
                    x_coordinates.clear()
                    y_coordinates.clear()
            else:
                associations.append(associations_for_one_item)
                associations_for_one_item = []
                continue
        associations.append(associations_for_one_item)
        # print(associations[0])
        return associations

    def write_out_all_merged_associations(self) -> None:
        self.__make_directory_for_merged_associations()
        with open(f'{self.out_data_path}/{self.out_data_merged_name}', 'w', newline='') as out_csvfile:
            if self.associations:
                # to avoid errors given by mess data, we can manually type fieldnames here later.
                writer = csv.DictWriter(out_csvfile, fieldnames=self.associations[0][0].keys())
                writer.writeheader()
                for item in self.associations:
                    for association_on_word in item:
                        writer.writerow(association_on_word)

    def sort_associations_by_itemid(self) -> None:
        #self.associations = sorted(self.associations, key=lambda x: (x[0]['expr_id'], x[0]['para_nr']))
        self.associations = sorted(self.associations, key=lambda x: int(x[0]['item_id']) if x and 'item_id' in x[0] else float('inf'))

    def _clear_noises_before_reading(self):
        for i in range(len(self.associations)):
            while self.associations[i] and (self.associations[i][0]['page_word_id'] not in self.start_sent):
                self.associations[i].pop(0)

    def write_out_denoise_merged_associations(self) -> None:
        self.__make_directory_for_merged_associations()
        self._clear_noises_before_reading()
        with open(f'{self.out_data_path}/{self.out_data_merged_denoise_name}', 'w', newline='') as out_csvfile:

            # to avoid errors given by mess data, we manually type fieldnames here later.
            fieldnames = ['sbm_id', 'stim_id', 'page_id', 'item_id', 'page_word_id', 'para_word_id', 'stim_word_id',
                          'exp_word_id', 'word', 'duration', 'start_t', 'end_t',
                          'x_mean', 'y_mean', 'response', 'rev', 'trial_id', 'difficulty']
            writer = csv.DictWriter(out_csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in self.associations:
                for association_on_word in item:
                    if self.threshold_lower < association_on_word['duration'] < self.threshold_upper and association_on_word['page_word_id'] is not None:
                        writer.writerow(association_on_word)

if __name__ == '__main__':
    from pathlib import Path
    input_folder = Path('../data/zh-word/divided')
    output_association_path = Path('../data/zh-word/associations/')
    annotation_path = Path('../data/zh-word/annotated/onestop_zh_annotation_words.csv')
    for file in input_folder.glob('*.csv'):
        print('I am identifying associations for :', file)
        obj_merger = associationMerger(file, output_association_path, 80, 4000, annotation_path)
        obj_merger.sort_associations_by_itemid()
        # obj_merger.write_out_all_merged_associations()
        obj_merger.write_out_denoise_merged_associations()
