'''
Functions for graphing accuracy plots and whatnot
'''
import matplotlib.pyplot as plt

def plot_history(adam_hist, sgd_hist=None, path="./", acc=True, loss=False):
	accs = list(adam_hist.history['acc'])
	val_accs = list(adam_hist.history['val_acc'])
	losses = list(adam_hist.history['loss'])
	val_losses = list(adam_hist.history['val_loss'])
	if sgd_hist:
		accs.extend(list(sgd_hist.history['acc']))
		val_accs.extend(list(sgd_hist.history['val_acc']))
		losses.extend(list(sgd_hist.history['loss']))
		val_losses.extend(list(sgd_hist.history['val_loss']))

	# Plot training & validation accuracy values
	plt.figure(1)
	plt.plot(accs)
	plt.plot(val_accs)
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(path + "accs.png",  bbox_inches='tight')

	# Plot training & validation loss values
	plt.figure(2)
	plt.plot(losses)
	plt.plot(val_losses)
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(path + "losses.png",  bbox_inches='tight')