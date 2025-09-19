from pathlib import Path
import pandas as pd
import os


class TrialDataPreprocessor:
    def __init__(self, raw_data_path: str, new_data_path: str):
        self.raw_data_path = Path(f"{raw_data_path}")
        self.new_data_path = Path(f"{new_data_path}")
        self.raw_data_df = pd.read_csv(self.raw_data_path, delimiter='\t', dtype={"experiment": str, "experiment_id": str,
                                                                                  "condition_id": str, "stimulus_id": str, "stimulus_name": str,
                                                                                  "page_id": str, "page_name": str, "item_id": str,
                                                                                  "question_id": str, "response_true": str})

    # Check if the directory for the divided files exists, if not, make one
    def _make_directory_for_divided_data(self) -> None:
        if not os.path.exists(self.new_data_path):
            os.mkdir(self.new_data_path)

    def split_sentence_into_words(self) -> None:
        self._make_directory_for_divided_data()
        # experiment_id is "zh1" for Chinese data
        self.raw_data_df['experiment_id'] = 'en1'
        self.raw_data_df = self.raw_data_df.\
            assign(text=self.raw_data_df['text'].str.split("@#@")).explode('text')
        self.raw_data_df['para_id'] = self.raw_data_df.groupby("item_id").cumcount()
        self.raw_data_df = self.raw_data_df.assign(text=self.raw_data_df['text'].str.split()).explode('text')
        self.raw_data_df['page_word_id'] = self.raw_data_df.groupby("item_id").cumcount()
        self.raw_data_df['para_word_id'] = self.raw_data_df.groupby(["item_id", "para_id"]).cumcount()
        self.raw_data_df['stim_word_id'] = self.raw_data_df.groupby("stimulus_id").cumcount()
        self.raw_data_df['exp_word_id'] = self.raw_data_df.groupby("experiment_id").cumcount()
        self.raw_data_df.to_csv(self.new_data_path / f'preprocessed_{self.raw_data_path.stem}.csv',
                                index=False, float_format=None)

    def filtered_new_df(self) -> None:
        df_new = self.raw_data_df[['experiment_id', 'condition_id', "stimulus_id", "page_id", 'item_id', 'para_id',
                                   'page_word_id', 'para_word_id', 'stim_word_id', 'exp_word_id', 'text', 'question_id', 'response_true']]
        df_new['word_nr'] = df_new.apply(lambda row: str(row['para_id']) + '-' + str(row['para_word_id']),
                                                  axis=1)
        df_new = df_new.rename(columns={'experiment_id': 'expr_id', 'condition_id': 'cond_id',
                                        'stimulus_id': 'stim_id', 'text': 'word'})
        df_new.to_csv(self.new_data_path / f'filtered_preprocessed_{self.raw_data_path.stem}.csv', index=False)

if __name__ == '__main__':
    trial_data_preprocessor = TrialDataPreprocessor('../data/en/trials/onestop_en.tsv', '../data/en/trials/')
    trial_data_preprocessor.split_sentence_into_words()
    trial_data_preprocessor.filtered_new_df()
