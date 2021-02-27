
import tensorflow as tf
import pandas
import numpy
from bootstrap.utils import make_feature_range, test_set_creator
from bootstrap.models import SimpleMod


def boots(data, feature, target, nodes, feature_range=0, boot_size, n_samples, test_splits, null = False):
	
	df = data.copy()

	x_axis = make_feature_range(feature_range, feature)
	
	frame = pd.DataFrame(0,index = range(n_samples), columns=x_axis)

	test_sets = test_set_creator(test_splits, df)


	for q in range(len(test_sets)):
    	print("---")
    	print(f"Fitting Test Set {q}")
    	test = df.iloc[test_sets[q],:]

    	for i in range(n_samples):
    		train_idxs = np.array([idx for idx in range(df.shape[0]) if idx not in test.index])

    		y = df[f'{target}']
    		X = df.loc[:, df.columns != f'{target}']

		    # Bootstrap #
		    boot_idxs = np.random.choice(train_idxs, size = boot_size, replace = True)

		    model = SimpleMod(output_dim=1,nodes=nodes)
		    model.compile(loss = 'mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))

		    history = model.fit(
		      	X.iloc[boot_idxs,:],
		      	y[boot_idxs],
		      	verbose=0,
		      	epochs=20)
		    if null:
				for j in range(len(x_axis)):
					newset = X.iloc[test.index,:].copy()
					newset[f'{feature}'] = np.random.choice(df[f'{feature}'], size = newset.shape[0],replace=False)
					pred = model.predict(newset.to_numpy())
					pred = np.mean(pred)
					frame.iloc[i:i+1,j:j+1] = pred

			if null == False:
				for j in range(len(x_axis)):
					newset = X.iloc[test.index,:].copy()
					newset[f'{feature}'] = x_axis[j]
					pred = model.predict(newset.to_numpy())
					pred = np.mean(pred)
					frame.iloc[i:i+1,j:j+1] = pred
	    if q == 0:
	      frames = frame + 0
	    if q > 0:
	      frames = frames + frame 

	return frame / (q+1)


