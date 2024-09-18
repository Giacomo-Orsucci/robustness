import matplotlib.pyplot as plt


def plotting(std_array, accuracy_array, psnr, x1_title, y1_title, y2_title, image_title):


    #std_array = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    #accuracy_array = [0.9972871895468235, 0.9614369208879388, 0.8513202080844546, 0.7599712079560951, 0.6991411828876849, 0.657818567729759, 0.628944142462253, 0.6068721658029914, 0.5907433247767806, 0.5773794022230629, 0.5667455349689067]
    #psnr = [0,10,20,30,40,50,60,70,80,90,100]


    #plt.plot(std_array, accuracy_array, marker='s', linestyle='--', color='black', markerfacecolor='red', markeredgecolor='red')
    #plt.grid(color='grey', linestyle='-', linewidth=0.5)



    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()


    PSNR, = ax2.plot(std_array, psnr, marker='s', linestyle='--', color='black', markerfacecolor='blue', markeredgecolor='blue', label='PSNR')
    Accuracy, = ax1.plot(std_array, accuracy_array, marker='s', linestyle='--', color='black', markerfacecolor='red', markeredgecolor='red', label='Accuracy')

    ax1.set_xlabel(x1_title)
    ax1.set_ylabel(y1_title)
    ax1.tick_params(axis="y")
    ax2.set_ylabel(y2_title)
    ax1.set_yticks([0.4,0.5,0.6,0.7,0.8,0.9,1.0]) #to fix the y scale but it can be used also accuracy_array


    #fig.legend(handles=[PSNR, Accuracy])

    # Create legend for both plots
    ax1.legend(handles=[PSNR, Accuracy], shadow=True, bbox_to_anchor=(0.05, -0.15), loc="lower right")
    ax1.grid(color='grey', linestyle='-', linewidth=0.5)


    plt.title(image_title, fontweight="bold")
    #fig.autofmt_xdate()
    plt.gca().invert_xaxis() #only for jpeg_compression
    plt.show()
