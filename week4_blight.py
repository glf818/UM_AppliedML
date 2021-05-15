import pandas as pd
import numpy as np

def blight_model():
    add_lat = pd.merge(pd.read_csv('addresses.csv'), 
             pd.read_csv('latlons.csv'), 
             left_on='address', right_on='address', 
             how='outer', suffixes=("","_y"))
    add_lat['lat'] = (add_lat['lat']-42.3898)*100
    add_lat['lon'] = (add_lat['lon']+83.1127)*100
    ticket_lat = dict(zip(add_lat['ticket_id'].values, add_lat['lat'].values))
    ticket_lon = dict(zip(add_lat['ticket_id'].values, add_lat['lon'].values))
    useless_feature_list = ['violator_name', 'zip_code', 'country', 'city',
                'inspector_name', 'violation_street_name',
                'violation_zip_code', 'violation_description',
                'mailing_address_str_number', 'mailing_address_str_name',
                'non_us_str_code', 'agency_name', 'state', 
                'ticket_issued_date', 'hearing_date', 'grafitti_status' ]
    dispList = ['Responsible by Default',
     'Responsible by Determination',
     'Responsible by Admission',
     'Responsible (Fine Waived) by Deter']
    #train_data = pd.read_csv('readonly/train.csv', encoding = 'ISO-8859-1')
    train_data = pd.read_csv('train.csv', encoding = 'ISO-8859-1')
    train_remove_list = [
            'balance_due',
            'collection_status',
            'compliance_detail',
            'payment_amount',
            'payment_date',
            'payment_status'
        ]
    train_data.drop(train_remove_list, inplace=True, axis=1)
    train_data = train_data.loc[lambda x: ~pd.isnull(x['compliance']),:] 
    train_data['InMi'] = \
    train_data['state'].map(lambda x: 1 if x=='MI' else 0)
    train_data['InDetroit'] = train_data['city'].str.lower().\
    map(lambda x: 2 if x=='detroit' else  1 if x =='southfield' else 0)
    train_data['lat'] = \
    train_data['ticket_id'].map(lambda x: ticket_lat.get(x,'1e5') )
    train_data['lon'] = \
    train_data['ticket_id'].map(lambda x: ticket_lon.get(x,'1e5'))
    train_data.drop(useless_feature_list, axis=1, inplace=True)
    train_data.dropna(how='any', inplace=True)
    train_data['violation_code'] = train_data['violation_code'].map(lambda x: x.split('-')[0] if x.find('-')>0 else '0')
    y_train = train_data.compliance
    X_train = train_data.drop('compliance', axis=1)
    X_train = pd.get_dummies(X_train)
    #test_data = pd.read_csv('readonly/test.csv')
    test_data = pd.read_csv('test.csv')
    test_data['InMi'] = \
    test_data['state'].map(lambda x: 1 if x=='MI' else 0)
    test_data['InDetroit'] = test_data['city'].str.lower().\
    map(lambda x: 2 if x=='detroit' else  1 if x =='southfield' else 0)
    test_data['lat'] = \
    test_data['ticket_id'].map(lambda x: ticket_lat.get(x,'1e5') )
    test_data['lon'] = \
    test_data['ticket_id'].map(lambda x: ticket_lon.get(x,'1e5'))
    test_data.drop(useless_feature_list, axis=1, inplace=True)
    test_data['lat'] = test_data['lat'].fillna(method='pad')
    test_data['lon'] = test_data['lon'].fillna(method='pad')
    test_data['disposition'] = test_data['disposition'].map(lambda x: x if x in dispList else 'Responsible by Default')
    test_data['violation_code'] = test_data['violation_code'].map(lambda x: x.split('-')[0] if x.find('-')>0 else '0')
    X_test = pd.get_dummies(test_data)
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=3)
    clf.fit(X_train, y_train)
   
    # Your code here
    res = pd.Series(clf.predict_proba(X_test)[:,1], index=test_data['ticket_id'])
    res.index.name='ticket_id'
    return  res# Your answer here
