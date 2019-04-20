import pandas as pd

l = ['marian', 'jane', '11', '11', 'loeki']


def check_for_name(l):

    def check(pos):
        while l[pos] == '11':
            pos -= 1
            check(pos)

        return l[pos]

    return [check(i) for i in range(len(l))]


# print(check_for_name(l))

# c = [(223.36141439205954, 966.483870967742), (222.45900601456157, 1097.1981639759417), (222.27279664250287, 1230.966170672771), (221.95266698702548, 1381.697741470447), (224.06125883540895, 807.317738135308), (230.29547894821008, 368.15454732369267), (288.2898936170213, 1099.4487681970884), (287.89502762430936, 1229.2696658774007), (287.62436660057284, 1383.7026878166996), (289.55, 967.3978873239437), (291.07484348321003, 806.0771200910643), (294.70412528845907, 384.07013768247464), (352.4520921357503, 1100.5296442687747), (351.69525873056335, 1231.091893958705), (351.37982994295555, 1385.6303950059198), (353.80250891932326, 968.2475543791), (355.5137166478767, 805.5186959789553), (358.2545874470191, 381.96088931753), (416.92777701496635, 1101.7095976932583), (416.0957528957529, 1232.2989703989704), (415.77753826255656, 1387.4177624488036), (418.54427173287274, 968.9386298215313), (420.35375291375294, 807.4150116550117), (422.4318308291597, 378.3075496197366), (479.9469261859179, 1389.03456711338), (481.30010819583447, 1102.2844197998377), (480.6354824165915, 1236.90261496844), (483.0724521380259, 969.4986816462226), (484.63103354240525, 808.4435051358968), (487.15982913792317, 365.59947919046095), (544.6291688447465, 1390.9282801881861), (546.5588705380927, 1103.0194459243473), (545.2227828939069, 1234.6305033430049), (548.3772943575799, 969.9542261500113), (549.7928988597386, 806.9534625011588), (551.4916804707395, 382.1898781120973), (609.4542953872196, 1392.1250528988573), (611.8567967698519, 1103.4467025572005), (609.9997440163829, 1234.7767822859337), (613.2624741319844, 970.2381007128075), (614.7532063372312, 807.0793096944549), (616.2869039760321, 380.1399989844107), (674.6695697252462, 1393.1228615863142), (675.4781738586616, 1236.0118824265166), (677.0854610398201, 1104.1468448207436), (678.542042380523, 970.706830477908), (679.9701741385699, 807.3015931826602), (681.5812113294056, 379.7171099662287), (740.343798894336, 1394.5149681860853), (741.194026040337, 1235.6903242277253), (742.8134258641398, 1104.433871613506), (744.1026892091229, 971.0836264609101), (745.4491191904048, 808.0773988005997), (747.3395977974623, 378.896768015322), (806.8830217281455, 1237.6123041940375), (805.7994021234924, 1395.6908566127204), (808.2321049840933, 1104.6712619300106), (809.4721497447532, 971.3107203630176), (810.5903962703962, 808.2561305361305), (812.4838663140177, 379.524332772301), (872.0098999795877, 1396.5456215554195), (874.1862771020624, 1105.2178741406663), (872.9603038470963, 1238.084293065425), (875.4179696351688, 971.5755721731249), (876.3917823534892, 808.5116929097177), (878.2926920023814, 378.3923397499504), (936.836028239581, 1392.1086312912776), (938.7897033158813, 1238.1896035901273), (940.0891488503429, 1105.3619739142127), (940.9533893428472, 971.5624352629761), (942.0431317393343, 809.5492733239569), (944.0418422414431, 381.4681254774402), (1003.6598709148653, 1398.1194549738757), (1004.1839611178615, 1239.7584447144593), (1005.2396301188903, 1106.4816380449142), (1006.9417197746375, 809.9178904590376), (1006.1198412698412, 972.5289115646259), (1009.4916913875021, 377.8688710416516), (1069.9430482367247, 1399.715545196595), (1071.3013609850939, 1107.4948801036942), (1070.212668593449, 1241.2208574181118), (1072.282555058028, 809.9460842547747), (1071.5610784422502, 973.6797958504383), (1075.0538417321327, 378.04138723893465), (1136.3512050653594, 1401.2136437908496), (1137.9672682434705, 974.6321040242125), (1137.437800597946, 1108.9626933575978), (1136.7128738621586, 1240.9512353706111), (1138.1859264041318, 810.204187033109), (1141.5291954868212, 373.6250430866098), (1203.7384162784997, 1243.7178229958324), (1202.9159440987453, 1402.2327858818728), (1207.8812171546547, 377.01032282282284), (1204.6228844738778, 810.4965967623252), (1204.3546067912841, 975.5368875124433), (1204.0795066911571, 1110.3404618210443), (1268.909171564822, 1398.657646667406), (1270.1598656382184, 1244.6277681015179), (1274.2329035663122, 380.9963557767222), (1271.130510814542, 811.3780027611597), (1270.9506048387098, 976.5847894265233), (1270.9292117057737, 1111.416029528078), (1336.7549554013874, 1245.5973736372646), (1336.9272292449607, 1397.6601753786586), (1337.9954828801156, 812.4180142740988), (1337.4445795115482, 977.2693115261355), (1337.2043881415698, 1112.1727830743112), (1341.2138873855856, 378.6286500858844), (1402.3213867914496, 1402.9512921408061), (1404.821190606226, 977.9634079737848), (1404.4313421256788, 1113.0859839668994), (1403.9327140346472, 1246.215164448908), (1405.4657436355517, 812.0085752568111), (1408.4369064399104, 378.401922499249), (1469.6136660380096, 1385.8019730161034), (1472.493940061701, 979.2453724107536), (1471.902566106552, 1114.0436368373062), (1470.962828236899, 1248.615437080823), (1473.4147686189444, 813.4317606652205), (1476.3474054921542, 371.2563971112696), (1537.5775945017183, 1390.889621993127), (1539.2912621359224, 1114.6692469168197), (1538.0271216097988, 1247.8697662792151), (1540.001223990208, 980.0580838989652), (1541.5036310820624, 813.5090777051562), (1543.8777206553596, 380.1386873588622), (1603.685338668191, 1390.4382680765932), (1607.1712813908564, 1115.3827430779138), (1605.0696640316205, 1247.8582015810277), (1609.3861457768132, 813.3810815699042), (1608.0333586050037, 980.6651142640528), (1611.8484302683478, 379.34629853400514), (1668.4665941578621, 1400.4507458048477), (1673.0686383240911, 1249.0576709796674), (1675.1550286607608, 1116.4399426784785), (1676.1002943420908, 981.4605908644936), (1677.3163649728936, 814.196269411008), (1679.3565144027962, 381.80633964083404), (1739.5600950118765, 1250.75296912114), (1743.0665632674163, 1117.3935633966653), (1744.005709361198, 982.0086179036949), (1744.7675915456207, 817.2679829520745), (1746.4453031045389, 370.445258563093), (1807.1322186314458, 1245.8615989883378), (1810.6448709463932, 1117.4113831899404), (1812.5884489647658, 815.8378132945877), (1811.963351363285, 982.1738602494756), (1813.6970916493133, 378.9843468494817), (1892.2639720294537, 640.5132348007276)]


def return_coords_table(coords):
    sorted_coords = sorted(coords, key=lambda x: [x[0][0], x[0][1]])

    all_cols = [sorted_coords[i::6][:24] for i in range(6)]

    data = {str(i): all_cols[i] for i in range(6)}
    df = pd.DataFrame(all_cols).T

    df = df[df.columns[::-1]]

    df.columns = ['Col1', 'Col2', 'col3', 'col4', 'col5', 'col6']
    # print(df)

    return df



