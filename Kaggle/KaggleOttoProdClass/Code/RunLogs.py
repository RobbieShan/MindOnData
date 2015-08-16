
>>> forest = rfc(n_estimators=100,n_jobs=-1)
>>> forest = forest.fit(nptrd[:,range(1,94)],nptrd[:,-1])
>>> ############################################
('PCA eExplained        ::', 1.0000000000000002)
('Training Error        ::', 1.0)
('Cross Validation Error::', 0.81185112634671885)
('Time Taken            ::', 590.9450690746307)
############################################



Gradient boosted Tree with 30 Learners and some starting values for variables that I typically find to work well in real-world datasets.

>>> forest = gbc(n_estimators=30, learning_rate=.1, min_samples_leaf=4,subsample=0.9,max_features=0.8,verbose=2,max_depth=10)
>>> forest = forest.fit(X,nptrd[:,-1])
      Iter       Train Loss      OOB Improve   Remaining Time 

         1       62705.3982        1653.4760           10.23m
         2       52988.1468         938.3360           10.06m
         3       45788.7096         664.1260            9.96m
         4       40174.3398         496.2680            9.77m
         5       35732.2012         390.8363            9.47m
         6       31962.4176         319.8060            9.12m
         7       28754.3878         254.9229            8.80m
         8       26074.9518         209.4890            8.41m
         9       23684.9956         175.0439            8.02m
        10       21781.9405         147.9455            7.63m
        11       19948.3880         127.2919            7.24m
        12       18301.4931         106.9417            6.87m
        13       16932.5998          86.6851            6.50m
        14       15614.4083          74.9941            6.13m
        15       14537.4320          63.2379            5.75m
        16       13475.3657          53.9234            5.37m
        17       12578.5109          46.9259            5.00m
        18       11726.1068          39.6631            4.62m
        19       10975.5635          35.1818            4.23m
        20       10228.4878          28.4253            3.85m
        21        9546.3791          21.7459            3.47m
        22        8974.4282          23.0070            3.09m
        23        8411.0833          19.9091            2.71m
        24        7913.9814          16.0347            2.32m
        25        7469.6653          13.1884            1.94m
        26        6993.9655          12.2990            1.55m
        27        6611.1292          10.7963            1.16m
        28        6229.3381           9.2066           46.59s
        29        5893.4688           5.7520           23.28s
        30        5568.0052           6.4390            0.00s
>>> ############################################
>>> ('PCA eExplained        ::', 1.0000000000000002)
>>> ('Training Error        ::', 0.98796372232138552)
>>> ('Cross Validation Error::', 0.78574926542605283)
>>> ('Time Taken            ::', 699.2878408432007)
>>> ############################################


The results show that the training data fit quite well at 98% accuracy but the cross-validation did not at 78% accuracy. 
So, the fit model probably has high-bias and has overfit. However, to confirm this, I tried with more trees in the model 
by bumping up the number of estimatores to 50 from the 30 that I started with originally.


>>> forest = gbc(n_estimators=50, learning_rate=.2, min_samples_leaf=4,subsample=0.9,max_features=0.8,verbose=2,max_depth=10)
>>> 
>>> forest = forest.fit(X,nptrd[:,-1])
      Iter       Train Loss      OOB Improve   Remaining Time 
         1       48802.1016        2954.5294           17.54m
         2       37539.9055        1018.8899           17.20m
         3       29953.5881         611.9287           17.09m
         4       24587.4078         418.6039           16.96m
         5       20595.5458         283.0042           16.64m
         6       17346.5840         195.0185           16.41m
         7       14939.3334         137.7900           16.11m
         8       12872.7913         101.9199           15.80m
         9       11158.1633          74.8559           15.50m
        10        9697.9818          52.1777           15.19m
        11        8656.7866          37.4963           14.81m
        12        7679.0980          32.4378           14.47m
        13        6796.4815          22.1986           14.11m
        14        6073.5736          16.1432           13.76m
        15        5494.2539          11.6250           13.36m
        16        4978.3135           7.6100           12.96m
        17        4462.4134           5.4640           12.57m
        18        4044.8705           6.1594           12.17m
        19        3675.9102           5.1824           11.79m
        20        3330.9701           2.2566           11.41m
        21        3094.3676           2.0988           10.99m
        22        2851.6525           0.8225           10.60m
        23        2654.9403          -4.0764           10.20m
        24        2450.7626           1.0755            9.79m
        25        2267.4626          -0.1766            9.39m
        26        2116.0157          -0.2629            8.99m
        27        1963.7995           0.1506            8.60m
        28        1837.4867           0.0388            8.20m
        29        1699.3624          -0.0602            7.82m
        30        1592.1313           0.4815            7.44m
        31        1482.7179          -0.5478            7.06m
        32        1390.8822          -0.5690            6.68m
        33        1296.9663           0.1603            6.29m
        34        1217.2286           0.3425            5.91m
        35        1142.6084          -0.4018            5.53m
        36        1057.7701          -0.5589            5.15m
        37         998.0345          -0.4233            4.77m
        38         929.1421          -0.7852            4.40m
        39         860.7627          -0.6619            4.02m
        40         799.7471          -0.6261            3.66m
        41         753.7884          -0.2681            3.28m
        42         699.3518          -0.3809            2.91m
        43         661.2513          -0.6953            2.54m
        44         622.8063          -0.3387            2.17m
        45         590.2156          -0.1967            1.81m
        46         549.1719          -0.4563            1.44m
        47         519.8277          -0.3430            1.08m
        48         488.8173           0.0092           43.02s
        49         462.2633          -0.5168           21.47s
        50         437.1182          -0.1563            0.00s
>>> ############################################
>>> ('PCA eExplained        ::', 1.0000000000000002)
>>> ('Training Error        ::', 1.0)
>>> ('Cross Validation Error::', 0.79035259549461312)
>>> ('Time Taken            ::', 1071.5469181537628)
>>> ############################################

This confirmed that addition more trees to the forest would lead to overfit. The training data fit 100% but the cross-validation data results did not improve
Now, I decided to reduce the number of trees in the model from 30 to 20 and also increased the learning rate from .1 to .25.

>>> forest = gbc(n_estimators=20, learning_rate=.25, min_samples_leaf=4,subsample=0.9,max_features=0.8,verbose=2,max_depth=10)

      Iter       Train Loss      OOB Improve   Remaining Time 
         1       43078.3306        3430.0816            6.81m
         2       32164.8434         966.7002            6.55m
         3       25100.0749         550.7592            6.24m
         4       20070.0918         333.5894            5.91m
         5       16432.4525         206.4025            5.58m
         6       13756.2590         136.0593            5.23m
         7       11632.9196          97.8728            4.88m
         8        9849.5939          54.3342            4.51m
         9        8532.2487          39.4402            4.14m
        10        7226.9000          23.8100            3.79m
        11        6288.2387          18.4429            3.41m
        12        5442.8444          11.2903            3.03m
        13        4865.7294           7.6504            2.65m
        14        4305.5489           5.2110            2.27m
        15        3822.9257           2.6044            1.88m
        16        3388.3615           1.2861            1.50m
        17        3067.8556           1.2703            1.13m
        18        2777.7776           2.0931           44.94s
        19        2541.0793          -1.8208           22.37s
        20        2271.5059          -1.8383            0.00s
>>> ############################################
>>> ('PCA eExplained        ::', 1.0000000000000002)
>>> ('Training Error        ::', 0.9990351681219547)
>>> ('Cross Validation Error::', 0.77717923604309502)
>>> ('Time Taken            ::', 1015.9430890083313)
>>> ############################################

The model still seemed to have high-bias. So, I wondered if this had anything to do with the fact that I had transformed my variables. 
Considering I was using all the principal components, I did not need to scale my predictors anymore. So, I decided to drop the pipelining 
for scaling and PCA and use the predictors as is without any transforms.

################# Without any Scaling ######################################################


>>> forest = gbc(n_estimators=15, learning_rate=.20, min_samples_leaf=4,subsample=0.9,max_features=0.8,verbose=2,max_depth=10)
>>> forest = forest.fit(nptrd[:,range(1,94)],nptrd[:,-1])
      Iter       Train Loss      OOB Improve   Remaining Time 
         1       50008.8029        3037.5342            2.45m
         2       39573.3551        1023.3383            2.31m
         3       32618.3630         631.8534            2.16m
         4       27802.2048         436.3276            2.00m
         5       24200.3489         301.2407            1.84m
         6       21474.1270         211.8438            1.67m
         7       19281.2996         157.7811            1.50m
         8       17467.9090         113.0769            1.33m
         9       15906.7029          83.7278            1.15m
        10       14624.4360          60.1180           57.78s
        11       13361.7413          53.1353           46.70s
        12       12461.7830          34.3530           35.21s
        13       11626.0341          32.3106           23.67s
        14       10830.1096          19.1302           11.87s
        15       10128.8550          18.1726            0.00s
>>> ############################################
>>> ('PCA eExplained        ::', 1.0000000000000002)
>>> ('Training Error        ::', 0.93294418447585503)
>>> ('Cross Validation Error::', 0.80127326150832512)
>>> ('Time Taken            ::', 505.007621049881)
>>> ############################################



>>> forest = gbc(n_estimators=15, learning_rate=.15, min_samples_leaf=4,subsample=0.9,max_features=0.8,verbose=2,max_depth=10)
>>> forest = forest.fit(nptrd[:,range(1,94)],nptrd[:,-1])
      Iter       Train Loss      OOB Improve   Remaining Time 
         1       56111.0604        2381.7322            2.56m
         2       45752.7435        1030.8509            2.36m
         3       38674.6762         675.1346            2.20m
         4       33612.2096         474.5899            2.02m
         5       29686.7297         350.4895            1.85m
         6       26604.5749         273.6163            1.66m
         7       24131.8794         201.2117            1.47m
         8       21962.1281         170.8479            1.30m
         9       20097.1981         133.7484            1.12m
        10       18522.1675         103.8954           56.56s
        11       17212.2572          86.0893           45.44s
        12       16010.1931          72.0427           34.24s
        13       15120.5555          59.2910           22.89s
        14       14137.2934          43.9850           11.48s
        15       13291.6022          40.3555            0.00s
>>> ############################################        14        8452.9637          35.4147            2.05m

>>> ('PCA eExplained        ::', 1.0000000000000002)
>>> ('Training Error        ::', 0.91234502387958893)
>>> ('Cross Validation Error::', 0.80034280117531831)
>>> ('Time Taken            ::', 173.61175990104675)
>>> ############################################




>>> forest = gbc(n_estimators=20, learning_rate=.20, min_samples_leaf=4,subsample=0.9,max_features=0.8,verbose=2,max_depth=10)
>>> forest = forest.fit(nptrd[:,range(1,94)],nptrd[:,-1])
      Iter       Train Loss      OOB Improve   Remaining Time 
         1       50108.5247        2976.6189            3.32m
         2       39564.9605        1029.1173            3.24m
         3       32677.6320         634.2885            3.11m
         4       27803.1616         435.6897            2.98m
         5       24149.8422         293.1369            2.89m
         6       21422.5899         209.5053            2.71m
         7       19230.4286         157.0459            2.56m
         8       17414.7089         112.7829            2.37mThis is the loss function used in (multinomial) logistic regression and extensions of it such as neural networks, defined as the negative log-likelihood of the true labels given a probabilistic classifier’s predictions. For a single sample with true label yt in {0,1} and estimated probability yp that yt = 1, the log loss is

         9       15798.3446          89.1855            2.19m
        10       14432.9783          65.1726            2.01m
        11       13440.8105          49.0754            1.81m
        12       12454.4345          38.7586            1.62m
        13       11605.2036          34.6549            1.43m
        14       10776.3554          23.2757            1.24m
        15       10093.7010          17.3839            1.03m
        16        9452.5174          16.0929           49.66s
        17        8879.3559           9.9281           37.44s
        18        8312.9216           5.2955           25.04s
        19        7806.1084           7.8743           12.59s
        20        7422.2547           0.2612            0.00s
>>> ############################################
>>> ('PCA eExplained        ::', 1.0000000000000002)
>>> ('Training Error        ::', 0.95496647209223795)
>>> ('Cross Validation Error::', 0.80626836434867777)
>>> ('Time Taken            ::', 555.5269808769226)
>>> ############################################





>>> forest = gbc(n_estimators=20, learning_rate=.20, min_samples_leaf=4,subsample=0.9,max_features=0.8,verbose=2,max_depth=5)
>>> forest = forest.fit(nptrd[:,range(1,94)],nptrd[:,-1])
      Iter       Train Loss      OOB Improve   Remain
         1       55800.9273        2625.1035           46.75s
         2       46886.7195         948.5774           42.14s
         3       41157.9956         634.6043           39.05s
         4       36967.6919         450.4196           36.76s
         5       33866.4161         318.1608           34.49s
         6       31524.7921         246.9721           32.72s
         7       29568.7327         190.4540           30.43s
         8       28091.9436         152.7077           28.05s
         9       26866.1646         114.6701           25.76s
        10       25761.0157          87.9198           23.46s
        11       25013.3031          70.8450           21.10s
        12       24245.5126          63.6605           18.83s
        13       23361.9013          58.0965           16.53s
        14       22758.6065          39.7026           14.17s
        15       22317.1884          35.4581           11.81s
        16       21816.0781          28.7020            9.43s
        17       21317.0349          28.5625            7.11s
        18       20990.4903          25.0188            4.74s
        19       20595.7180          26.2498            2.37s
        20       20230.4376          18.8346            0.00s
>>> ############################################
>>> ('PCA eExplained        ::', 1.0000000000000002)
>>> ('Training Error        ::', 0.81805682859761686)
>>> ('Cross Validation Error::', 0.78741429970617038)
>>> ('Time Taken            ::', 680.0386788845062)
>>> ############################################




>>> forest = gbc(n_estimators=20, learning_rate=.20, min_samples_leaf=4,subsample=0.9,max_features=0.8,verbose=2,max_depth=7)
>>> forest = forest.fit(nptrd[:,range(1,94)],nptrd[:,-1])
      Iter       Train Loss      OOB Improve   Remaining Time 
         1       53268.5298        2857.9043            1.46m
         2       43637.5154         991.0578            1.40m
         3       37489.9569         618.4841            1.34m
         4       33148.5900         427.8989            1.26m
         5       29832.9667         314.6905            1.20m
         6       27280.7158         232.5823            1.12m
         7       25248.9209         173.8457            1.05m
         8       23706.6736         130.4872           58.43s
         9       22293.1689         106.0485           53.70s
        10       21069.4253          78.7714           49.21s
        11       20031.2644          66.1939           44.43s
        12       19184.4607          52.2250           39.40s
        13       18314.1613          44.2523           34.        14        8452.9637          35.4147            2.05m
73s
        14       17697.5812          38.8250           29.74s
        15       17166.7534          26.3979           24.79s
        16       16554.3338          29.5106           19.84s
        17       15943.2076          19.3726           14.86s
        18       15555.6923          15.5751            9.92s
        19       15109.4216          11.9085            4.95s
        20       14740.7691           7.5998            0.00s
>>> ############################################forest = gbc(n_estimators=20, learning_rate=.20, min_samples_leaf=5,subsample=.9,max_features=0.8,verbose=2,max_depth=8)

forest = forest.fit(nptrd[:,range(1,94)],nptrd[:,-1])
#forest = forest.fit(X,nptrd[:,-1])

temp = forest.predict(nptrd[:,range(1,94)])
#temp = forest.predict(X)
TrainError = sum(temp == nptrd[:,-1]) / (len(nptrd)*1.0)

# Need to spend some time checking for overfit - using some elbow techiques maybe


# Cross validate the model using the cross validation dataset
This is the loss function used in (multinomial) logistic regression and extensions of it such as neural networks, defined as the negative log-likelihood of the true labels given a probabilistic classifier’s predictions. For a single sample with true label yt in {0,1} and estimated probability yp that yt = 1, the log loss is

#XCv = pipeline.transform(npcvd[:,range(1,94)])
outputCv = forest.predict(npcvd[:,range(1,94)])
#outputCv = forest.predict(XCv)
CrossValidError = sum(outputCv == npcvd[:,-1]) / (len(npcvd)*1.0)

end = time.time()

print('############################################')
print('PCA eExplained        ::', PCAExplained)
print('Training Error        ::', TrainError)
print('Cross Validation Error::', CrossValidError)
print('Time Taken            ::', end - start )
print('############################################')
>>> ('PCA eExplained        ::', 1.0000000000000002)
>>> ('Training Error        ::', 0.87519899657484679)
>>> ('Cross Validation Error::', 0.80063663075416258)
>>> ('Time Taken            ::', 800.2680690288544)
>>> ############################################


>>> forest = gbc(n_estimators=20, learning_rate=.20, min_samples_leaf=4,subsample=1,max_features=0.8,verbose=2,max_depth=8)
>>> forest = forest.fit(nptrd[:,range(1,94)],nptrd[:,-1])
      Iter       Train Loss   Remaining Time 
         1       58040.1499            2.18m
         2       46890.2594            2.06m
         3       39779.4614            1.96m
         4       34607.4869            1.87m
         5       30826.2837            1.76m
         6       27981.6017            1.65m
         7       25633.9041            1.54m
         8       23733.0097            1.42m
         9       22159.1246            1.31m
        10       20786.6395            1.19m
        11       19609.3795            1.08m
        12       18591.7858           57.70s
        13       17687.2600           50.46s
        14       16806.4431           43.43s
        15       16092.2253           36.27s
        16       15373.1711           29.08s
        17       14768.6044           21.88s
        18       14174.9323           14.65s
        19       13683.4024            7.30s
        20       13218.5286            0.00s
>>> ############################################
>>> ('PCA eExplained        ::', 1.0000000000000002)
>>> ('Training Error        ::', 0.90918519947899079)
>>> ('Cross Validation Error::', 0.80597453476983349)
>>> ('Time Taken            ::', 983.7646250724792)
>>> ############################################





>>> forest = gbc(n_estimators=20, learning_rate=.20, min_samples_leaf=5,subsample=.9,max_features=0.8,verbose=2,max_depth=10)
>>> forest = forest.fit(nptrd[:,range(1,94)],nptrd[:,-1])
      Iter       Train Loss      OOB Improve   Remaining Time 
         1       50235.5222        3084.6407            3.34m
         2       39685.4422        1033.0641            3.18m
         3       32814.3804         615.5104            3.08m
         4       28074.1888         423.5680            2.91m
         5       24478.1295         307.9707            2.73m
         6       21796.0290         207.7437            2.54m
         7       19534.7336         152.3102            2.39m
         8       17665.4682         122.9958            2.21m
         9       16171.1918          88.8121            2.05m
        10       14904.0793          64.2790            1.88m
        11       13729.9069          56.4068            1.71m
        12       12710.4902          33.6134            1.53m
        13       11882.5390          28.2066            1.34m
        14       11092.9862          20.6577            1.16m
        15       10449.2114          15.0696           58.19s
        16        9869.5763           9.7603           46.55s
        17        9316.6393          10.9828           35.02s
        18        8670.8970           6.1131           23.49s
        19        8267.4421           6.3402           11.75s
        20        7796.2739           2.2569            0.00s
>>> ############################################
>>> ('PCA eExplained        ::', 1.0000000000000002)
>>> ('Training Error        ::', 0.95113126537700809)
>>> ('Cross Validation Error::', 0.80798237022526931)
>>> ('Time Taken            ::', 236.70602297782898)
>>> ############################################




>>> forest = gbc(n_estimators=20, learning_rate=.15, min_samples_leaf=5,subsample=.9,max_features=0.8,verbose=2,max_depth=10)
>>> forest = forest.fit(nptrd[:,range(1,94)],nptrd[:,-1])
      Iter       Train Loss      OOB Improve   Remaining Time 
         1       56294.8473        2423.6619            3.30m
         2       46017.6688        1035.4225            3.16m
         3       39096.8381         676.1957            2.99m
         4       33963.9792         477.0696            2.83m
         5       30054.4925         355.9261            2.67m
         6       26958.7617         273.3757            2.50m
         7       24318.2716         206.9316            2.34m
         8       22253.9009         168.5955            2.18m
         9       20354.6459         134.2762            2.01m
        10       18938.9591         108.4656            1.84m
        11       17539.8139          87.4627            1.66m
        12       16395.9651          66.9301            1.49m
        13       15328.6105          61.5379            1.31m
        14       14451.1461          46.7845            1.13m
        15       13613.7935          39.9700           56.60s
        16       12858.1480          32.3921           45.47s
        17       12109.2621          24.5864           34.27s
        18       11520.4815          22.9186           22.92s
        19       10934.4130          16.8241           11.52s
        20       10373.2572          18.7153            0.00s
>>> ############################################
>>> ('PCA eExplained        ::', 1.0000000000000002)
>>> ('Training Error        ::', 0.9312074870953736)
>>> ('Cross Validation Error::', 0.80783545543584723)
>>> ('Time Taken            ::', 232.47739911079407)
>>> ############################################


>>> forest = gbc(n_estimators=20, learning_rate=.15, min_samples_leaf=6,subsample=.9,max_features=0.8,verbose=2,max_depth=10)
>>> forest = forest.fit(nptrd[:,range(1,94)],nptrd[:,-1])
      Iter       Train Loss      OOB Improve   Remaining Time 
         1       56546.8912        2392.6771            3.34m
         2       46107.9678        1037.2610            3.20m
         3       39145.1405         661.6398            3.07m
         4       34082.7947         480.7448            2.91m
         5       30188.5954         355.0120            2.74m
         6       26990.7851         262.8160            2.57m
         7       24495.3719         214.0475            2.40m
         8       22344.9374         168.1545            2.23m
         9       20569.1785         124.9220            2.04m
        10       19100.9824         104.0449            1.86m
        11       17752.8264          85.2098            1.69m
        12       16500.8772          67.8861            1.51m
        13       15511.9188          62.1695            1.33m
        14       14578.7322          43.3950            1.15m
        15       13718.5064          41.0221           57.73s
        16       12840.5714          28.9768           46.33s
        17       12293.4693          24.1105           34.80s
        18       11588.5469          21.1679           23.27s
        19       11066.8874          18.4426           11.66s
        20       10532.2169          12.4979            0.00s
>>> ############################################
>>> ('PCA eExplained        ::', 1.0000000000000002)
>>> ('Training Error        ::', 0.93094215832891114)
>>> ('Cross Validation Error::', 0.80636630754162586)
>>> ('Time Taken            ::', 234.32576513290405)
>>> ############################################



>>> forest = gbc(n_estimators=20, learning_rate=.15, min_samples_leaf=6,subsample=.9,max_features=0.6,verbose=2,max_depth=15)
>>> forest = forest.fit(nptrd[:,range(1,94)],nptrd[:,-1])
      Iter       Train Loss      OOB Improve   Remaining Time 
         1       53680.9333        2483.3204            5.82m
         2       42427.1640        1060.3164            5.76m
         3       34746.5872         679.9189            5.44m
         4       29146.9550         478.5820            5.17m
         5       24854.2744         356.0554            4.84m
         6       21418.6090         266.5871            4.56m
         7       18674.9793         202.3411            4.29m
         8       16441.4918         153.2908            3.98m
         9       14483.1150         120.6584            3.66m
        10       12925.9653          98.3893            3.33m
        11       11588.7612          71.4932            3.01m
        12       10324.2902          59.9403            2.69m
        13        9368.8537          41.8094            2.37m
        15        7642.7121          25.8918            1.71m
        16        6930.4676          22.1377            1.38m
        17        6321.1942          16.0385            1.05m
        18        5758.9983          12.1138           42.21s
        19        5211.6780           9.6812           21.24s
        20        4814.4728           7.6748            0.00s
>>> ############################################
>>> ('PCA eExplained        ::', 1.0000000000000002)
>>> ('Training Error        ::', 0.98441796516956925)
>>> ('Cross Validation Error::', 0.81229187071498532)
>>> ('Time Taken            ::', 427.1028461456299)
>>> ############################################




