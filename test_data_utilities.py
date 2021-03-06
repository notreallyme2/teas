from data.skl_synthetic import *
home = Path.home()
path_for_data = home/"teas-data/sklearn/"

def test_skl_dataloader():
    """Test that the data loaded by load_skl_data (and therefore the data generated by make_skl_dataset) 
    is correct and won't break things later on
    """
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = load_skl_data(path_for_data)
    # check shape
    assert X_train.shape == (6000, 100)
    assert X_valid.shape == (2000, 100)
    assert X_test.shape == (2000, 100)
    assert Y_train.shape == (6000, 1000)
    assert Y_valid.shape == (2000, 1000)
    assert Y_test.shape == (2000, 1000)
    # check values
    test_array = np.array([-1.99951767e-04, -9.27996893e-05, -2.59568397e-03, -2.09976888e-03,
        6.58589880e-04, -4.23221372e-04,  1.92915897e-03,  1.49961230e-03,
        1.26562198e-03,  1.01843179e-03,  2.71854337e-03, -1.43129020e-04,
        5.41678644e-04,  6.84339067e-05, -4.18649228e-04, -6.13575802e-04,
       -2.34344981e-05, -1.00931277e-03, -1.89829694e-03, -9.29253612e-04,
       -1.13770553e-03,  1.08772548e-03,  7.11305041e-04, -2.20094936e-03,
       -3.86251895e-05, -5.58333186e-04, -8.27023444e-04,  4.68814888e-04,
        1.53396455e-03,  3.06722355e-04, -3.02796990e-04, -2.41960811e-03,
        1.45011001e-03, -1.87866582e-03, -1.45386074e-05,  4.14929838e-04,
       -1.66910298e-03,  2.03757115e-04, -1.04608906e-03,  4.03851159e-03,
        8.95163585e-04,  1.05970051e-03, -1.08579730e-03,  1.93777773e-03,
       -2.06261417e-03, -5.51351423e-04,  1.05811671e-03, -8.23039906e-05,
        1.13866462e-04,  1.31397803e-03, -8.45309284e-04, -2.08699127e-03,
       -7.30026935e-04,  2.14783240e-03, -3.12534581e-03, -1.31111522e-03,
       -1.02198876e-03,  5.54193959e-04,  2.66602141e-04, -8.68263136e-04,
        1.57713057e-03,  5.61367586e-04,  4.42794333e-04,  1.96350261e-03,
       -5.00949612e-04,  1.27474956e-03,  1.38906630e-03,  1.93386429e-03,
       -5.90645551e-04, -1.93303177e-03, -5.07912778e-04, -7.19980845e-04,
        7.55118156e-05,  2.28959121e-03, -3.82255500e-04,  1.21291996e-03,
       -1.43115104e-04,  1.30432749e-03, -2.05476493e-03, -1.23082944e-03,
        6.61768683e-04, -1.94758418e-04,  7.35187721e-04, -1.98385463e-03,
        4.77706078e-04,  4.35981264e-04, -7.77568697e-04, -1.62564894e-03,
       -2.93187371e-03,  2.57595540e-03, -4.45171494e-04, -2.27905509e-04,
       -1.02543635e-03, -1.40736880e-03, -7.97401685e-04,  6.02963205e-05,
        3.62766780e-04, -1.44818805e-03,  4.35283647e-05, -8.12319614e-06])
    assert all([np.isclose(a, b, atol=1e-10) for a, b in zip(X_train[0], test_array)])