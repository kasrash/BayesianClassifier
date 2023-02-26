import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# you can comment the following two lines if not interested in using the styles
import scienceplots
plt.style.use(['science','no-latex'])

def sample_plot_3d(dataset, bins, fignum, plot_path):
    '''
    Generate 3D plot of a given 2D samples by generating their histograms.

    Input:
    dataset = samples with shape (samples, features)
    bins = number of bins to generate the histogram
    fignum = figure number
    plot_path = location to save the plots

    Return:
    None.
    '''
    # calculate the histogram of the given dataset
    hist, xedges, yedges = np.histogram2d(dataset[:,0], dataset[:,1], bins=bins, density=True)
    # create a mesh for plotting
    X, Y = np.meshgrid(xedges[:-1] + 0.5 * (xedges[1:] - xedges[:-1]), yedges[:-1] + 0.5 * (yedges[1:] - yedges[:-1]))
    fig = plt.figure(fignum, figsize=(7, 5))
    # plot the histogram in 3d
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    mappable.set_array(hist)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, hist, cmap=mappable.cmap)
    ax.set_xlabel("$X_1$")
    ax.set_ylabel("$X_2$")
    ax.set_zlabel("Probability")
    plt.savefig(plot_path+f'\samplesplot_{fignum}.png', dpi=600)

def sample_plot_2d(dataset, bins, fignum, plot_path):
    '''
    Generate 2D plot of a given 2D samples by generating their histograms.

    Input:
    dataset = samples with shape (samples, features)
    bins = number of bins to generate the histogram
    fignum = figure number
    plot_path = location to save the plots

    Return:
    None.
    '''
    # calculate the histogram of the given dataset
    hist, xedges, yedges = np.histogram2d(dataset[:,0], dataset[:,1], bins=bins, density=True)
    # create a mesh for plotting
    X, Y = np.meshgrid(xedges[:-1] + 0.5 * (xedges[1:] - xedges[:-1]), yedges[:-1] + 0.5 * (yedges[1:] - yedges[:-1]))
    fig = plt.figure(fignum, figsize=(7, 5))
    ax = plt.axes()
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    mappable.set_array(hist)
    ax.imshow(hist, cmap=mappable.cmap, norm=mappable.norm, extent=(np.min(X), np.max(X), np.min(Y), np.max(Y)), interpolation='none')
    ax.set_xlabel("$X_1$")
    ax.set_ylabel("$X_2$")
    cbar = plt.colorbar(mappable)
    cbar.set_label("Probability")
    plt.savefig(plot_path+f'\samplesplot_{fignum}.png', dpi=600)

def db_plot(dataset, dataset_slicer, hp, fignum, plot_path, hp2=None):
    '''
    Generate a plot of a given 2D samples and the decision boundary from given hyperplanes.

    Input:
    dataset = samples with shape (samples, features)
    dataset_slicer = where to seperate data to seperate samples from dist. 1 from dist. 2
    hp = hyperplanes points
    fignum = figure number
    plot_path = location to save the plots
    hp2 = second hyperplanes for comparison

    Return:
    None.
    '''
    plt.figure(fignum, figsize=(7,5))
    #plot samples from dist. 1
    plt.scatter(dataset[:dataset_slicer,0],dataset[:dataset_slicer,1],c='b', label="Dist. 1 Samples")
    # plot samples from dist. 2
    plt.scatter(dataset[dataset_slicer:,0],dataset[dataset_slicer:,1],c='r',alpha=0.7, label="Dist. 2 Samples")
    #plot decision boundary
    plt.contour(hp[1], hp[2], hp[0], levels=[0], colors='k', linewidths=3)  # decision boundary is where hp is zero
    plt.plot([],[],'k',label="Decision Boundary", lw=3)
    if hp2:
        plt.plt.contour(hp2[1], hp2[2], hp2[0], levels=[0], colors='g', linewidths=3)  # decision boundary is where hp is zero
        plt.plot([],[],'g',label="Decision Boundary #2", lw=3)
    plt.legend(loc=1)
    plt.xlim(np.min(hp[1]), np.max(hp[1])+1)
    plt.ylim(np.min(hp[2]), np.max(hp[2])+1)
    plt.savefig(plot_path+'\decisionbound.png', dpi=600)

def conf_matrix_plot(conf_matrix, fignum, plot_path, labels=[1,2], normalize=False):
    '''
    Generate a plot of a given confusion matrix.

    Input:
    conf_matrix = confusion matrix
    fignum = figure number
    plot_path = location to save the plots
    labels = labels to be used for plotting the confusion matrix.
    nrmalize = option to plot the confusion matrix normalized row-wise (class-wise)

    Return:
    None.
    '''
    fig = plt.figure(fignum, figsize=(7,5))
    ax = plt.axes()
    #normalize the confusion matrix
    if normalize:
        conf_matrix = conf_matrix.astype('float32')
        conf_matrix[0,:] /= np.sum(conf_matrix[0,:])
        conf_matrix[1,:] /= np.sum(conf_matrix[1,:])
    #plot confusion matrix
    cm_plot = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)
    cm_plot.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    if normalize:
        plt.savefig(plot_path+'\confusionmatrix_norm.png', dpi=600)
    else:
        plt.savefig(plot_path+'\confusionmatrix.png', dpi=600)
