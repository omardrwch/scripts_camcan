"""
Save a file, births.json, containing a dictionary 'birth' such that:
birth[subject] = year of the subject's birth
"""

import pandas

def get_ages(subjects = ['CC110033', 'CC110037']):
    """
    Return a list containing the ages of the subjects in the list 'subjects'.
    """

    # Load file
    df = pandas.read_csv('participants.csv')

    ages = []
    for subject in subjects:
        age = df[df.Observations == subject]['age'].values[0]
        ages.append(age)

    return ages

if __name__ == '__main__':
    pass
    # subject = 'CC510220'
    # raw = camcan_utils.get_raw(subject, 'rest')
    #df = pandas.read_csv('participants.csv')
