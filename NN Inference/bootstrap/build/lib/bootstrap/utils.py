import pandas
import numpy

def data_grabber(data):

	if data == 'white_wine':
		path = "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/master/winequality-white.csv"
		df = pd.read_csv(path,delimiter=";")

	if data == 'red_wine':
		path = "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/master/winequality-red.csv"
		df = pd.read_csv(path,delimiter=";")

	return df


# not optimized in the least, but very straightforward
def scale(data,center=False,zero_one=False,mean_zero=False):
  df = data.copy()

  if center:
    for i in range(df.shape[1]):
      df.iloc[:,i] = (df.iloc[:,i] - np.mean(df.iloc[:,i])) / np.std(df.iloc[:,i])
    return df

  if zero_one:
    for i in range(df.shape[1]):
      df.iloc[:,i] = (df.iloc[:,i] - np.min(df.iloc[:,i])) / (np.max(df.iloc[:,i])- np.min(df.iloc[:,i]))
    return df

  if mean_zero:
    for i in range(df.shape[1]):
      df.iloc[:,i] = 2*(df.iloc[:,i] - np.min(df.iloc[:,i])) / (np.max(df.iloc[:,i])- np.min(df.iloc[:,i])) - 1
    return df

def make_feature_range(num_range,feature):
	
 	if num_range == 0:
 		x_axis = np.linspace(
 			np.min(data[f'{feature}']),
 			np.max(data[f'{feature}']),
 			num = 20)
 	
 	if num_range > 0:
 		x_axis = np.linspace(
 			np.min(data[f'{feature}']) - np.std(data[f'{feature}'])*num_range, 
 			np.max(data[f'{feature}']) + np.std(data[f'{feature}'])*num_range,
 			num = 20)

 	return x_axis


def test_set_creator(splits,data):

	test_sets = []
	already_used = []
	df = data.copy()

	indexs = list(range(df.shape[0]))
	size = round(1/splits,2)

	for k in range(splits):
	    if k == 0:
	      test_idxs = np.random.choice(indexs, size = int(size*df.shape[0]), replace = False)
	      already_used += list(test_idxs)
	      test_sets.append(test_idxs)

	    if k > 0:
	      indexs = [idx for idx in indexs if idx not in already_used]
	      test_idxs = np.random.choice(indexs, size = int(size*df.shape[0]), replace = False)
	      already_used += list(test_idxs)
	      test_sets.append(test_idxs)
	      
	      if k+1 == len(range(splits)):
	        indexs = [idx for idx in indexs if idx not in already_used]
	        test_sets[-1] = np.append(test_sets[-1],indexs)

	return test_sets





