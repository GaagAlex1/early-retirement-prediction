import pandas as pd
from catboost import CatBoostClassifier

def create_submission(test_df: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    loaded_model = CatBoostClassifier()
    loaded_model.load_model('best.cbm') # Загрузка модели по весам

    erly_pnsn = pd.DataFrame(loaded_model.predict(test_df), columns=['erly_pnsn_flg']) #Предсказание
    submission = pd.concat([accounts, erly_pnsn], axis=1) #Производим конкатенацию предсказанных значений и accnt_id

    return submission


if __name__ == '__main__':
    test_df = pd.read_csv('prep_test.csv', sep=';', encoding='cp1251') #Загрузка обработанного тестового датасета
    accounts = pd.read_csv('raw_test.csv', sep=';', encoding='cp1251', usecols=['accnt_id']) #Загрузка accnt_id

    submission_df = create_submission(test_df, accounts)
    submission_df.to_csv('submission.csv', index=False, encoding='utf-8') #Сохранение DataFrame в csv