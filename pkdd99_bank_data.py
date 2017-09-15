# -*- coding: utf-8 -*-
'''
Read/process/provide/document PKDD99 bank transactions data.

This module reads, processes, provides, and documents an anonymized, public bank
transactions dataset from the Czech Republic in 1993-1998 - this was the shared
dataset for the PKDD99 Challenge, from the European conference series:

"European Conferences on Machine Learning and
 European Conferences on Principles and Practice of Knowledge Discovery
  in Databases ECML/PKDD Discovery Challenges 1999 - 2005 :
  A Collaborative Effort in Knowledge Discovery from Databases"

This module does not provide the original data itself; you must download that,
unzip it, and place its eight ASCII files in the current working directory.
(See link to original conference datafile in references below.)
The dataset description document describing the fields and format is also
available at the original conference website (again see references below).

These data are anonymized bank transactions covering six years of bank
transactions for 5369 bank clients with 4500 accounts (ie some of those clients
are coupled on some of the accounts).  The currency, the koruna (crown) or CZK,
varied in the 25-30/USD range in the 1990s during the dataset.

The dataset is not in English, and contained in csv files, so this function
reads the csv files, translates and reorganizes the PKDD99 data into a set of
Pandas dataframes all in English, and provides some limited documentation of
the dataset based on the data description document.

References
----------
PKDD99 dataset zip file: http://sorry.vse.cz/~berka/challenge/pkdd1999/data_berka.zip
PKDD99 data description document: http://sorry.vse.cz/~berka/challenge/pkdd1999/berka.htm
PKDD99 conference website: http://sorry.vse.cz/~berka/challenge/PAST

Author:  Andrew Ganse, http://research.ganse.org, (unaffiliated with original dataset)

'''

import pandas as pd


def get_bank_data():
    '''
    Read, process, and provide the PKDD99 bank transactions data.

    Parameters
    ----------
    (none)

    Returns
    -------
    account, card, client, disp, district, loan, order, trans : Pandas dataframe
        The translated contents of the original PKDD99 dataset, contained in
        Panads dataframes.  Further details are available in meta-data attached
        to these dataframes in the .notes and .description attributes, e.g.
        account.notes.  Additionally the loan dataframe has a meta-data
        attribute of codes, i.e. loan.codes, returning the status definitions.
        The rest of the details are in the dataset's original data description
        document in the references.

    Notes
    -----
    Assumes the files account.asc, card.asc, client.asc, disp.asc, district.asc,
    loan.asc, order.asc, trans.asc from the original data distribution exist in
    the current working directory.

    Examples
    --------
    >>> import pkdd99_bank_data as pkdd99
    >>> account,card,client,disp,district,loan,order,trans = pkdd99.get_bank_data()

    References
    ----------
    Original data description document with further info about fields/format:
    http://sorry.vse.cz/~berka/challenge/pkdd1999/berka.htm

    '''
    # FIXME: ideally should have a check first that these files exist...
    account = pd.read_csv('account.asc','delimiter',';')
    card = pd.read_csv('card.asc','delimiter',';')
    client = pd.read_csv('client.asc','delimiter',';')
    disp = pd.read_csv('disp.asc','delimiter',';')
    district = pd.read_csv('district.asc','delimiter',';')
    loan = pd.read_csv('loan.asc','delimiter',';')
    order = pd.read_csv('order.asc','delimiter',';')
    trans = pd.read_csv('trans.asc','delimiter',';',low_memory=False)

    account.name = 'Account'
    card.name = 'Card'
    client.name = 'Client'
    disp.name = 'Disp'
    district.name = 'District'
    loan.name = 'Loan'
    order.name = 'Order'
    trans.name = 'Trans'

    # descriptions are cut/pasted from the Financial Data Description webpage for the data
    account.description = 'each record describes static characteristics of an account'
    card.description = 'each record describes a credit card issued to an account'
    client.description = 'each record describes characteristics of a client'
    disp.description = 'each record relates together a client with an account'
    district.description = 'each record describes demographic characteristics of a district'
    loan.description = 'each record describes a loan granted for a given account'
    order.description = 'each record describes characteristics of a payment order'
    trans.description = 'each record describes one transaction on an account'

    account.notes = '(one account can have one or more clients, e.g. married couples)'
    card.notes = '(one account can have one or more credit cards)'
    client.notes = '(one client can have one or more accounts)'
    disp.notes = '(disposition connects a client/account pair and allows to link one or more cards)'
    district.notes = '(neighborhoods for both bank/account branches and client homes.  same 16 fields as original A1-16.)'
    loan.notes = '(one account may have zero or one loan.  see loan.codes for ABCD status definitions)'
    order.notes = '(one payment order is from one account)'
    trans.notes = '(category,bank,account are NaN for some types/operations.)'
    loan.codes = 'Loan status codes:\nA = contract finished, no problems\nB = contract finished, loan not payed\nC = running contract, OK so far\nD = running contract, client in debt'

    account,card,client,disp,district,loan,order,trans = _translate_and_clean(
        account,card,client,disp,district,loan,order,trans
    )

    return account,card,client,disp,district,loan,order,trans


def table_summary(df):
    '''
    Output a brief summary for one of the eight PKDD99 dataframes.

    Parameters
    ----------
    df : Pandas dataframe
        One of the eight dataframes output from get_bank_data().

    Returns
    -------
    (no returned objects - outputs text strings to stdout)

    Examples
    --------
    >>> import pkdd99_bank_data as pkdd99
    >>> account,card,client,disp,district,loan,order,trans = pkdd99.get_bank_data()
    >>> pkdd99.table_summary(account)

    Account : length 4500 : each record describes static characteristics of an account
    (one account can have one or more clients, e.g. married couples)
       account_id  district_id stmt_frq       date
    0         576           55  monthly 1993-01-01
    1        3818           74  monthly 1993-01-01
    2         704           55  monthly 1993-01-01

    '''

    print(df.name,': length',len(df),':',df.description)
    print(df.notes)
    print(df.head(3))
    print(' ')


def _translate_and_clean(account,card,client,disp,district,loan,order,trans):
    # Basic preprocessing / initial feature engineering:
    # Translate from Czech to English, undo funny formats like for gender, add a few agg stats...

    def num2date(x):
        if isinstance(x, str):
            return pd.to_datetime('19'+x, format='%Y%m%d %H:%M:%S')
        else:
            return pd.to_datetime(str(float(x)+19000000.), format='%Y%m%d')


    # Account:
    account['date'] = account['date'].apply(lambda x: num2date(x))
    account['frequency'].replace('POPLATEK MESICNE','monthly',inplace=True)
    account['frequency'].replace('POPLATEK TYDNE','weekly',inplace=True)
    account['frequency'].replace('POPLATEK PO OBRATU','after_tr',inplace=True) # after transaction
    account.rename(columns = {'frequency':'stmt_frq'}, inplace=True) # statement freq
    # Card:
    card['issued'] = card['issued'].apply(lambda x: num2date(x))
    card.rename(columns = {'issued':'date'}, inplace=True) # date credit card issued
    # Client:
    client['MM']=client['birth_number']//100 - client['birth_number']//10000*100
    client['gender'] = 'M'
    client.loc[client['MM']>50,'gender'] = 'F'
    client.loc[client['gender']=='F','birth_number'] -= 5000
    client['birth_number'] = client['birth_number'].apply(lambda x: num2date(x))
    client.rename(columns = {'birth_number':'date_birth'}, inplace=True) # client's birthdate
    client.drop('MM',1,inplace=True)
    # Disp:
    disp['type'].replace('OWNER','owner',inplace=True)
    disp['type'].replace('DISPONENT','disponent',inplace=True)
    # District:
    district.rename(columns = {
    'A1':'district_id','A2':'dname','A3':'region','A4':'pop','A5':'nmu500','A6':'nmu2k',
    'A7':'nmu10k','A8':'nmuinf','A9':'ncit','A10':'rurba','A11':'avgsal',
    'A12':'urat95','A13':'urat96','A14':'ent_ppt','A15':'ncri95','A16':'ncri96'}, inplace=True)
    # Loan:
    loan['date'] = loan['date'].apply(lambda x: num2date(x))
    # Order:
    order['k_symbol'].replace('POJISTNE','ins_paymt',inplace=True) # insurrance payment
    order['k_symbol'].replace('SIPO','household',inplace=True)
    order['k_symbol'].replace('LEASING','leasing',inplace=True)
    order['k_symbol'].replace('UVER','loan_payt',inplace=True) # loan payment
    order.rename(columns = {'k_symbol':'category'}, inplace=True)
    # Trans  # takes ~5min on my mbp w/ 16gb ram
    trans['date'] = trans['date'].apply(lambda x: num2date(x))
    trans['type'].replace('PRIJEM','credit',inplace=True)
    trans['type'].replace('VYDAJ','withdrawal',inplace=True)
    trans['operation'].replace('VYBER KARTOU','creditcard_wd',inplace=True) # credit card withdrawal
    trans['operation'].replace('VKLAD','credit_in_cash',inplace=True)
    trans['operation'].replace('PREVOD Z UCTU','coll_from_bank',inplace=True) # collection from another bank
    trans['operation'].replace('VYBER','cash_wd',inplace=True) # cash withdrawal
    trans['operation'].replace('PREVOD NA UCET','remi_to_bank',inplace=True) # remittance to another bank
    trans['k_symbol'].replace('POJISTNE','ins_paymt',inplace=True) # insurrance payment
    trans['k_symbol'].replace('SLUZBY','paymt_for_stmt',inplace=True) # payment for statement(?)
    trans['k_symbol'].replace('UROK','int_credited',inplace=True) # interest credited
    trans['k_symbol'].replace('SANKC. UROK','sanc_int',inplace=True) # sanction interest for neg balance
    trans['k_symbol'].replace('SIPO','household',inplace=True)
    trans['k_symbol'].replace('DUCHOD','pension',inplace=True) # old-age pension
    trans['k_symbol'].replace('UVER','loan_paymt',inplace=True) # loan payment
    trans.rename(columns = {'k_symbol':'category'}, inplace=True)
    # Note the snafu that pandas int columns can't contain NaNs, but the
    # trans.account (destination account#) field does, so it's type float.
    # But account_id column is always filled so it's an integer.

    return account,card,client,disp,district,loan,order,trans
