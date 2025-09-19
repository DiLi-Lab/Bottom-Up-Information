from pathlib import Path
import pandas as pd
import numpy as np
import os
from collections import defaultdict


class FeatureExtractor:
    def __init__(self, input_data_path_trial, input_data_path_associations,
                 output_data_path):
        # It is not necessary to set attribute 'association_threshold' because we have set it in `mergeassociations`, but I'd
        # like to keep it here for the flexibility of working with existing association files and for the convenience of testing.
        self.input_path_trial = input_data_path_trial
        self.input_path_associations = input_data_path_associations
        self.input_df_f = pd.read_csv(self.input_path_associations, na_values=['NA', ''])
        self.input_df_f['index'] = range(len(self.input_df_f))

        self.output_path = output_data_path
        self.output_df = pd.read_csv(self.input_path_trial, na_values=['NA'])
        self.output_name_stem = self.input_path_associations.stem.split('_')[1]

        self.input_df_ff = self.input_df_f.loc[self.input_df_f['duration'] > 0]

############## for Chinese words processing ##################
        # self.output_df.drop(columns=[
        #     'word', 'page_word_id', 'para_word_id',
        #     'stim_word_id', 'exp_word_id', 'word_nr', 'rank'
        # ], inplace=True, errors='ignore')
        # self.output_df.drop_duplicates(inplace=True)
        # self.output_df.rename(columns={
        #     'tokens': 'word',
        #     'page_token_id': 'page_word_id',
        #     'para_token_id': 'para_word_id',
        #     'stim_token_id': 'stim_word_id',
        #     'exp_token_id': 'exp_word_id'
        # }, inplace=True)

    def _make_directory(self) -> None:
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

    def __fill_mode(self, series):
        mode = series.mode()
        if not mode.empty:
            return series.fillna(mode.iloc[0])
        else:
            return series

    def extra_info(self):
        """
        Get extra information about the experiment and merge it into the output dataframe
        """
        extra_info_df = self.input_df_f[['stim_id', 'page_id', 'item_id', 'page_word_id', 'rev', 'difficulty', 'trial_id']].drop_duplicates()
        self.output_df = self.output_df.merge(extra_info_df, on=['stim_id', 'page_id', 'item_id', 'page_word_id'], how='left')
        # fill in the nan values in 'trial_id' with the majority value of the 'trial_id' in the same 'item_id'
        self.output_df['trial_id'] = self.output_df.groupby('item_id')['trial_id'].transform(self.__fill_mode)
        # convert trial_id from float to int
        self.output_df['trial_id'] = self.output_df['trial_id'].fillna(-99).astype(int)
        self.output_df['difficulty'] = self.output_df.groupby('item_id')['difficulty'].transform(self.__fill_mode)
        self.output_df['rev'] = self.output_df.groupby('item_id')['rev'].transform(self.__fill_mode)
        self.output_df['question_id'] = self.output_df['question_id'].fillna(-99).astype(int)
        self.output_df['response_true'] = self.output_df['response_true'].fillna('--')

    def get_first_duration(self):
        """
        Calculate the first association duration for each word and merge it into the output dataframe.
        """
        input_df_f_grouped = self.input_df_ff.groupby(['stim_id', 'page_id', 'item_id','page_word_id'])
        first_association_values = input_df_f_grouped['duration'].first()
        first_association_df = pd.DataFrame(first_association_values).reset_index()
        first_association_df = first_association_df.rename(columns={'duration': 'first_duration'})
        self.output_df = self.output_df.merge(first_association_df, on=['stim_id', 'page_id', 'item_id','page_word_id'], how='left')
        self.output_df['first_duration'] = self.output_df['first_duration'].fillna(0)

    def get_total_duration(self):
        """
        Calculate the total association duration for each word and merge it into the output dataframe
        """
        input_df_f_grouped = self.input_df_ff.groupby(['stim_id', 'page_id', 'item_id','page_word_id'])
        total_duration_values = input_df_f_grouped['duration'].sum()
        total_duration_df = pd.DataFrame(total_duration_values).reset_index()
        total_duration_df = total_duration_df.rename(columns={'duration': 'total_duration'})
        self.output_df = self.output_df.merge(total_duration_df, on=['stim_id', 'page_id', 'item_id','page_word_id'], how='left')
        self.output_df['total_duration'] = self.output_df['total_duration'].fillna(0)

    # gaze duration equals to first-pass reading time: sum of the durations of all first-pass associations on a word
    # (0 if the word was skipped in the first-pass)
    # here our IA is a word, gaze duration also equals to first association
    def get_gaze_duration(self):
        gaze_duration_dict = defaultdict(int)
        input_df_ff_grouped = self.input_df_ff.groupby(['stim_id', 'page_id', 'item_id'])
        gaze_duration_df = pd.DataFrame()
        for name, group in input_df_ff_grouped:
            for i in group.index:
                # if a word behind it has been focused, its gaze duration is 0
                if (group.loc[:i-1, 'page_word_id'] < group.loc[i, 'page_word_id']).all():
                    gaze_duration_dict[group.loc[i, 'page_word_id']] += group.loc[i, 'duration']
            gaze_duration = pd.DataFrame(
                {
                 'stim_id': name[0],
                 'page_id': name[1],
                 'item_id': name[2],
                 'page_word_id': gaze_duration_dict.keys(),
                 'gaze_duration': gaze_duration_dict.values()}).sort_values(by='page_word_id')
            gaze_duration_df = pd.concat([gaze_duration_df, gaze_duration], ignore_index=True)
            gaze_duration_dict.clear()
        self.output_df = self.output_df.merge(gaze_duration_df, on=['stim_id', 'page_id', 'item_id','page_word_id'], how='left')
        self.output_df['gaze_duration'] = self.output_df['gaze_duration'].fillna(0)

    # sum of all first pass association durations on a word until a word to the right of this word is fixated
    #  0 if the word was not fixated in the first-pass
    def get_right_bounded_rt(self):
        rrt_dict = defaultdict(int)
        # input_df_ff_grouped = self.input_df_ff.groupby(['para_nr'])
        # input_df_ff_grouped = self.input_df_ff.groupby(['expr_id', 'para_nr'])
        input_df_ff_grouped = self.input_df_ff.groupby(['stim_id', 'page_id', 'item_id'])
        rrt_df = pd.DataFrame()
        for name, group in input_df_ff_grouped:
            for i in group.index:
                if (group.loc[:i, 'page_word_id'] <= group.loc[i, 'page_word_id']).all():
                    rrt_dict[group.loc[i, 'page_word_id']] += group.loc[i, 'duration']
            rrt = pd.DataFrame(
                {
                 'stim_id': name[0],
                 'page_id': name[1],
                 'item_id': name[2],
                 'page_word_id': rrt_dict.keys(),
                 'right_bounded_rt': rrt_dict.values()}).sort_values(by='page_word_id')
            rrt_df = pd.concat([rrt_df, rrt], ignore_index=True)

            rrt_dict.clear()
        self.output_df = self.output_df.merge(rrt_df, on=['stim_id', 'page_id', 'item_id','page_word_id'], how='left')
        self.output_df['right_bounded_rt'] = self.output_df['right_bounded_rt'].fillna(0)

    #  sum of all association durations starting from the first first-pass association on a word until association a word to the
    #  right of this word (including all regressive associations on previous words),
    #  0 if the word was not fixated in the first-pass
    def get_go_past_time(self):
        gpt_dict = defaultdict(int)
        input_df_ff_grouped = self.input_df_ff.groupby(['stim_id', 'page_id', 'item_id'])
        gpt_df = pd.DataFrame()
        for name, group in input_df_ff_grouped:
            for i in group.index:
                if (group.loc[:i-1, 'page_word_id'] < group.loc[i, 'page_word_id']).all():
                    gpt_dict[group.loc[i, 'page_word_id']] += group.loc[i, 'duration']
                    for index in group.loc[i+1:, 'index']:
                        if group.loc[index, 'page_word_id'] <= group.loc[i, 'page_word_id']:
                            gpt_dict[group.loc[i, 'page_word_id']] += group.loc[index, 'duration']
                        else:
                            break
            gpt = pd.DataFrame(
                {
                 'stim_id': name[0],
                 'page_id': name[1],
                 'item_id': name[2],
                 'page_word_id': gpt_dict.keys(),
                 'go_past_time': gpt_dict.values()}).sort_values(by='page_word_id')
            gpt_df = pd.concat([gpt_df, gpt], ignore_index=True)
            gpt_dict.clear()
        self.output_df = self.output_df.merge(gpt_df, on=['stim_id', 'page_id', 'item_id','page_word_id'], how='left')
        self.output_df['go_past_time'] = self.output_df['go_past_time'].fillna(0)

    def check_comprehension_answer(self):
        """
        Check whether the comprehension question is answered correctly or not. Correct --> 1; Wrong --> 0
        """
        input_df_ff_grouped = self.input_df_ff.groupby(['stim_id', 'page_id', 'item_id', 'page_word_id'])
        solution_to_check = input_df_ff_grouped['response'].first()
        solution_df = pd.DataFrame(solution_to_check).reset_index()
        solution_df = solution_df.rename(columns={'response': 'response_chosen'})
        self.output_df = self.output_df.merge(solution_df, on=['stim_id', 'page_id', 'item_id', 'page_word_id'], how='left')
        solution_to_fill = self.output_df.groupby(['stim_id', 'page_id', 'item_id'])['response_chosen'].first()
        solution_to_fill_df = pd.DataFrame(solution_to_fill).reset_index()
        self.output_df = self.output_df.drop(columns='response_chosen').merge(solution_to_fill_df,
                                                                              on=['stim_id', 'page_id', 'item_id'], how='left')
        self.output_df['response_chosen'] = self.output_df['response_chosen'].fillna('--')
        self.output_df['correctness'] = self.output_df.apply(
            lambda x: int(1) if str(x['response_chosen']).strip() == str(x['response_true']).strip() else int(0), axis=1)
        # if 'response_chosen' is '--', correctness is -99
        self.output_df['correctness'] = self.output_df.apply(
            lambda x: int(-99) if str(x['response_chosen']) == '--' else x['correctness'], axis=1)

    def get_binary(self):
        """
        Get binary variable FPFix: whether a word is fixed in the first-pass reading, 1, fixed, 0, otherwise;
        FPReg: First Pass Regression Out --> whether there is a regression initiated from a word, 1, yes, 0, no.
        RegIn_incl: whether there is a regression landed on a word, 1, yes, 0, no, including all regressions
        RegIn_excl: whether there is a regression landed on a word, 1, yes, 0, no, only including regressions on words that were fixed in the first pass
        """
        FPFix_dict = defaultdict(int)
        FPReg_dict = defaultdict(int)
        RegIn_incl_dict = defaultdict(int)
        RegIn_excl_dict = defaultdict(int)
        input_df_ff_grouped = self.input_df_ff.groupby(['stim_id', 'page_id', 'item_id'])
        binary_df = pd.DataFrame()
        for name, group in input_df_ff_grouped:
            for i in group.index:
                if (group.loc[:i, 'page_word_id'] <= group.loc[i, 'page_word_id']).all():
                    FPFix_dict[group.loc[i, 'page_word_id']] = 1
                    for index in group.loc[i+1:, 'index']:
                        if group.loc[index, 'page_word_id'] < group.loc[i, 'page_word_id']:
                            FPReg_dict[group.loc[i, 'page_word_id']] = 1
                        else:
                            break

            # Calculate RegIn_incl and RegIn_excl
            for i in group.index:
                current_page_word_id = group.loc[i, 'page_word_id']
                for j in group.loc[:i-1].index:
                    if group.loc[j, 'page_word_id'] > current_page_word_id:
                        # RegIn_incl: Include all regressions
                        RegIn_incl_dict[current_page_word_id] = 1
                        # RegIn_excl: Only include if the word was fixed in the first pass
                        if int(FPFix_dict.get(current_page_word_id, 0)) == 1:
                            RegIn_excl_dict[current_page_word_id] = 1
                        break

            FPFix = pd.DataFrame(
                {
                 'stim_id': name[0],
                 'page_id': name[1],
                 'item_id': name[2],
                 'page_word_id': FPFix_dict.keys(),
                 'FPFix': FPFix_dict.values()}).sort_values(by='page_word_id')
            FPReg = pd.DataFrame(
                {
                 'stim_id': name[0],
                 'page_id': name[1],
                 'item_id': name[2],
                 'page_word_id': FPReg_dict.keys(),
                 'FPReg': FPReg_dict.values()}).sort_values(by='page_word_id')
            RegIn_incl = pd.DataFrame({
                'stim_id': name[0],
                'page_id': name[1],
                'item_id': name[2],
                'page_word_id': RegIn_incl_dict.keys(),
                'RegIn_incl': RegIn_incl_dict.values()}).sort_values(by='page_word_id')
            RegIn_excl = pd.DataFrame({
                'stim_id': name[0],
                'page_id': name[1],
                'item_id': name[2],
                'page_word_id': RegIn_excl_dict.keys(),
                'RegIn_excl': RegIn_excl_dict.values()}).sort_values(by='page_word_id')
            FPFix = FPFix.merge(FPReg, on=['stim_id', 'page_id', 'item_id','page_word_id'], how='left')
            FPFix = FPFix.merge(RegIn_excl, on=['stim_id', 'page_id', 'item_id','page_word_id'], how='left')
            FPFix = FPFix.merge(RegIn_incl, on=['stim_id', 'page_id', 'item_id','page_word_id'], how='outer')
            binary_df = pd.concat([binary_df, FPFix], ignore_index=True)
            FPFix_dict.clear()
            FPReg_dict.clear()
            RegIn_incl_dict.clear()
            RegIn_excl_dict.clear()
        self.output_df = self.output_df.merge(binary_df, on=['stim_id', 'page_id', 'item_id','page_word_id'], how='left')
        self.output_df['FPFix'] = self.output_df['FPFix'].fillna(0).astype(int)
        self.output_df['FPReg'] = self.output_df['FPReg'].fillna(0).astype(int)
        self.output_df['RegIn_incl'] = self.output_df['RegIn_incl'].fillna(0).astype(int)
        self.output_df['RegIn_excl'] = self.output_df['RegIn_excl'].fillna(0).astype(int)

    def get_NFix(self):
        """
        Calculate the total number of fixations for each word and merge it into the output dataframe
        """
        input_df_f_grouped = self.input_df_ff.groupby(['stim_id', 'page_id', 'item_id', 'page_word_id'])
        number_fixations = input_df_f_grouped['duration'].count()
        number_fixations_df = pd.DataFrame(number_fixations).reset_index()
        number_fixations_df = number_fixations_df.rename(columns={'duration': 'NFix'})
        self.output_df = self.output_df.merge(number_fixations_df, on=['stim_id', 'page_id', 'item_id', 'page_word_id'], how='left')
        self.output_df['NFix'] = self.output_df['NFix'].fillna(0).astype(int)

    def write_out(self):
        """
        Write all the above reading measures out to a csv file.
        """
        self._make_directory()
        filtered_df = self.output_df[self.output_df['trial_id'] != -99]
        filtered_df.to_csv(self.output_path / f'reader_{self.output_name_stem}_reading_measures.csv', index=False)


if __name__ == '__main__':
    from pathlib import Path

    input_fixation_folder = Path('/Users/cui/Documents/uzh/PhD/Projects/Re-Veil/Re-Veil/post_processing/data/zh-chr/associations')
    input_fixation_paths = input_fixation_folder.glob('*.csv')
    input_trial_path = Path('/Users/cui/Documents/uzh/PhD/Projects/Re-Veil/Re-Veil/post_processing/data/zh-chr/annotated/onestop_zh_annotation_chr.csv')
    output_rt_path = Path('/Users/cui/Documents/uzh/PhD/Projects/Re-Veil/Re-Veil/post_processing/data/zh-chr/rm-zh-chr')

    for input_fixation_path in input_fixation_paths:
        print("I am computing reading measures for : ", input_fixation_path)
        obj_feature_extractor = FeatureExtractor(input_trial_path,
                                            input_fixation_path,
                                            output_rt_path)
        if not obj_feature_extractor.input_df_f.empty:
            obj_feature_extractor.extra_info()
            obj_feature_extractor.check_comprehension_answer()
            obj_feature_extractor.get_first_duration()
            obj_feature_extractor.get_total_duration()
            obj_feature_extractor.get_gaze_duration()
            # obj_feature_extractor.get_right_bounded_rt()
            obj_feature_extractor.get_go_past_time()
            obj_feature_extractor.get_binary()
            obj_feature_extractor.get_NFix()
            obj_feature_extractor.write_out()
