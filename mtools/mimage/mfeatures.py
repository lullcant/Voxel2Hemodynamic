

import SimpleITK as sitk
import radiomics.featureextractor as FEE


def get_radiomics_features(image_path, label_path, param_path=None):
    '''
    Extract features (based on pyradiomics)
    ----------------------------------------------------------------------------------------------------------------
    ## Gray Level Cooccurence Matrix (GLCM) [24]               ## first order                            [19]
    - 'Autocorrelation'                     ## 1.              - 'Energy'                                ## 1.
    - 'JointAverage'                        ## 2.              - 'TotalEnergy'                           ## 2.
    - 'ClusterProminence'                   ## 3.              - 'Entropy'                               ## 3.
    - 'ClusterShade'                        ## 4.              - 'Minimum'                               ## 4.
    - 'ClusterTendency'                     ## 5.              - '10Percentile'                          ## 5.
    - 'Contrast'                            ## 6.              - '90Percentile'                          ## 6.
    - 'Correlation'                         ## 7.              - 'Maximum'                               ## 7.
    - 'DifferenceAverage'                   ## 8.              - 'Mean'                                  ## 8.
    - 'DifferenceEntropy'                   ## 9.              - 'Median'                                ## 9.
    - 'DifferenceVariance'                  ## 10.             - 'InterquartileRange'                    ## 10.
    - 'JointEnergy'                         ## 11.             - 'Range'                                 ## 11.
    - 'JointEntropy'                        ## 12.             - 'MeanAbsoluteDeviation'                 ## 12.
    - 'Imc1'                                ## 13.             - 'RobustMeanAbsoluteDeviation'           ## 13.
    - 'Imc2'                                ## 14.             - 'RootMeanSquared'                       ## 14.
    - 'Idm'                                 ## 15.             - 'StandardDeviation'                     ## 15.(disable)
    - 'MCC'                                 ## 16.             - 'Skewness'                              ## 16.
    - 'Idmn'                                ## 17.             - 'Kurtosis'                              ## 17.
    - 'Id'                                  ## 18.             - 'Variance'                              ## 18.
    - 'Idn'                                 ## 19.             - 'Uniformity'                            ## 19.
    - 'InverseVariance'                     ## 20.
    - 'MaximumProbability'                  ## 21.
    - 'SumAverage'                          ## 22. (repeat)
    - 'SumEntropy'                          ## 23.
    - 'SumSquares'                          ## 24.
    ----------------------------------------------------------------------------------------------------------------
    ## Gray Level Run Length Matrix (GLRLM) [16]               ## Gray Level Size Zone Matrix (GLSZM)    [16]
    - 'ShortRunEmphasis'                    ## 1.              - 'SmallAreaEmphasis'                     ## 1.
    - 'LongRunEmphasis'                     ## 2.              - 'LargeAreaEmphasis'                     ## 2.
    - 'GrayLevelNonUniformity'              ## 3.              - 'GrayLevelNonUniformity'                ## 3.
    - 'GrayLevelNonUniformityNormalized'    ## 4.              - 'GrayLevelNonUniformityNormalized'      ## 4.
    - 'RunLengthNonUniformity'              ## 5.              - 'SizeZoneNonUniformity'                 ## 5.
    - 'RunLengthNonUniformityNormalized'    ## 6.              - 'SizeZoneNonUniformityNormalized'       ## 6.
    - 'RunPercentage'                       ## 7.              - 'ZonePercentage'                        ## 7.
    - 'GrayLevelVariance'                   ## 8.              - 'GrayLevelVariance'                     ## 8.
    - 'RunVariance'                         ## 9.              - 'ZoneVariance'                          ## 9.
    - 'RunEntropy'                          ## 10.             - 'ZoneEntropy'                           ## 10.
    - 'LowGrayLevelRunEmphasis'             ## 11.             - 'LowGrayLevelZoneEmphasis'              ## 11.
    - 'HighGrayLevelRunEmphasis'            ## 12.             - 'HighGrayLevelZoneEmphasis'             ## 12.
    - 'ShortRunLowGrayLevelEmphasis'        ## 13.             - 'SmallAreaLowGrayLevelEmphasis'         ## 13.
    - 'ShortRunHighGrayLevelEmphasis'       ## 14.             - 'SmallAreaHighGrayLevelEmphasis'        ## 14.
    - 'LongRunLowGrayLevelEmphasis'         ## 15.             - 'LargeAreaLowGrayLevelEmphasis'         ## 15.
    - 'LongRunHighGrayLevelEmphasis'        ## 16.             - 'LargeAreaHighGrayLevelEmphasis'        ## 16.
    ----------------------------------------------------------------------------------------------------------------
    ## Shape Features (3D)                  [17]               ## Gray Level Dependence Matrix (GLDM)    [14]
    - 'MeshVolume'                          ## 1.              - 'SmallDependenceEmphasis'               ## 1.
    - 'VoxelVolume'                         ## 2.              - 'LargeDependenceEmphasis'               ## 2.
    - 'SurfaceArea'                         ## 3.              - 'GrayLevelNonUniformity'                ## 3.
    - 'SurfaceVolumeRatio'                  ## 4.              - 'DependenceNonUniformity'               ## 4.
    - 'Sphericity'                          ## 5.              - 'DependenceNonUniformityNormalized'     ## 5.
    - 'Compactness1'                        ## 6. (disabled)   - 'GrayLevelVariance'                     ## 6.
    - 'Compactness2'                        ## 7. (disabled)   - 'DependenceVariance'                    ## 7.
    - 'SphericalDisproportion'              ## 8. (disabled)   - 'DependenceEntropy'                     ## 8.
    - 'Maximum3DDiameter'                   ## 9.              - 'LowGrayLevelEmphasis'                  ## 9.
    - 'Maximum2DDiameterSlice'              ## 10.             - 'HighGrayLevelEmphasis'                 ## 10.
    - 'Maximum2DDiameterColumn'             ## 11.             - 'SmallDependenceLowGrayLevelEmphasis'   ## 11.
    - 'Maximum2DDiameterRow'                ## 12.             - 'SmallDependenceHighGrayLevelEmphasis'  ## 12.
    - 'MajorAxisLength'                     ## 13.             - 'LargeDependenceLowGrayLevelEmphasis'   ## 13.
    - 'MinorAxisLength'                     ## 14.             - 'LargeDependenceHighGrayLevelEmphasis'  ## 14.
    - 'LeastAxisLength'                     ## 15.
    - 'Elongation'                          ## 16.
    - 'Flatness'                            ## 17.
    ----------------------------------------------------------------------------------------------------------------
    Neighbouring Gray Tone Difference Matrix (NGTDM)
    - 'Coarseness'                          ## 1.
    - 'Contrast'                            ## 2.
    - 'Busyness'                            ## 3.
    - 'Complexity'                          ## 4.
    - 'Strength'                            ## 5.
    ----------------------------------------------------------------------------------------------------------------
    :param param_path: setting path
    :param image_path: image path
    :param label_path: label path (usually tumor)
    :return: feature, key
    '''

    ## 使用配置文件初始化特征抽取器
    extractor = FEE.RadiomicsFeatureExtractor()
    if param_path != None:
        extractor.loadParams(param_path)
    else:
        extractor.enableAllFeatures()

    ## 抽取特征
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)

    # image = normalizeImage(image)
    result = extractor.execute(image, label)

    ## 特征与其对应的关键词
    features = dict()
    keys = []

    # 输出特征
    for key, value in result.items():
        features[key] = value
        keys.append(key)

    return features, keys


def translate_keys():
    '''
    ## Gray Level Cooccurence Matrix (GLCM)  ## First Order Statistic               ## Shape 3D
    glcm.autocorrelation                     fos.energy                             shape3d.mesh_volume
    glcm.joint_average                       fos.total_energy                       shape3d.voxel_volume
    glcm.cluster_prominence                  fos.entropy                            shape3d.surface_area
    glcm.cluster_shade                       fos.minimum                            shape3d.surfaceVolumeRatio
    glcm.cluster_tendency                    fos.10th_percentile                    shape3d.sphericity
    glcm.contrast                            fos.90th_percentile                    shape3d.max_3d_diameter
    glcm.correlation                         fos.maximum                            shape3d.max_2d_diameter
    glcm.difference_average                  fos.mean                               shape3d.max_2d_diameter_column
    glcm.difference_entropy                  fos.median                             shape3d.max_2d_diameter_row
    glcm.difference_variance                 fos.interquartile_range                shape3d.major_axis_length
    glcm.joint_energy                        fos.range                              shape3d.minor_axis_length
    glcm.joint_entropy                       fos.mean_absolute_deviation            shape3d.least_axis_length
    glcm.imc1                                fos.robust_mean_absolute_deviation
    glcm.imc2                                fos.root_mean_squared
    glcm.idm                                 fos.skewness
    glcm.mcc                                 fos.kurtosis
    glcm.idmn                                fos.variance
    glcm.id                                  fos.uniformity
    glcm.idn
    glcm.inverse_variance
    glcm.max_probability
    glcm.sum_entropy
    glcm.sum_squares
    -------------------------------------------------------------------------------------------------------------------
    ##Gray Level Dependence Matrix (GLDM)                      ## Gray Level Run Length Matrix (GLRLM)
    gldm.small_dependence_emphasis                             glrlm.short_run_emphasis
    gldm.large_dependence_emphasis                             glrlm.long_run_emphasis
    gldm.gray_level_non_uniformity                             glrlm.gray_level_non_uniformity
    gldm.dependence_non_uniformity                             glrlm.gray_level_non_uniformity_normalied
    gldm.dependence_non_uniformity_normalized                  glrlm.run_length_non_uniformity
    gldm.gray_level_variance                                   glrlm.run_length_non_uniformity_normalized
    gldm.dependence_variance                                   glrlm.run_percentage
    gldm.dependence_entropy                                    glrlm.gray_level_variance
    gldm.low_gray_level_emphasis                               glrlm.run_variance
    gldm.high_gray_level_emphasis                              glrlm.run_entropy
    gldm.small_dependence_low_gray_level_emphasis              glrlm.low_gray_level_run_empathsis
    gldm.small_dependence_high_gray_level_emphasis             glrlm.hight_gray_level_run_empathsis
    gldm.large_dependence_low_gray_level_emphasis              glrlm.short_run_low_gray_level_emphasis
    gldm.large_dependence_high_gray_level_emphasis             glrlm.short_run_high_gray_level_emphasis
    -------------------------------------------------------------------------------------------------------------------
    ## ray Level Size Zone Matrix (GLSZM)                ## Neighbouring Gray Tone Difference Matrix (NGTDM)
    glszm.small_area_emphasis                            ngtdm.coarseness
    glszm.large_area_emphasis                            ngtdm.contrast
    glszm.gray_level_non_uniformity                      ngtdm.busyness
    glszm.gray_level_non_uniformity_normalized           ngtdm.complexity
    glszm.size_zone_non_uniformity                       ngtdm.strength
    glszm.size_zone_non_uniformity_normalized
    glszm.zone_percentage
    glszm.gray_level_variance
    glszm.zone_variance
    glszm.zone_entropy
    glszm.low_gray_level_zone_emphasis
    glszm.high_gray_level_zone_emphasis
    glszm.small_area_low_gray_level_emphasis
    glszm.small_area_high_gray_level_emphasis
    glszm.large_area_low_gray_level_emphasis
    glszm.large_area_high_gray_level_emphasis

    :param keys:
    :return:
    '''

    features = {
        ########################### Shape ################################
        "shape3d.mesh_volume": "original_shape_MeshVolume",
        "shape3d.voxel_volume": "original_shape_VoxelVolume",
        "shape3d.surface_area": "original_shape_SurfaceArea",
        "shape3d.surfaceVolumeRatio": "original_shape_SurfaceVolumeRatio",
        "shape3d.sphericity": "original_shape_Sphericity",
        "shape3d.max_3d_diameter": "original_shape_Maximum3DDiameter",
        "shape3d.max_2d_diameter": "original_shape_Maximum2DDiameterSlice",
        "shape3d.max_2d_diameter_column": "original_shape_Maximum2DDiameterColumn",
        "shape3d.max_2d_diameter_row": "original_shape_Maximum2DDiameterRow",
        "shape3d.major_axis_length": "original_shape_MajorAxisLength",
        "shape3d.minor_axis_length": "original_shape_MinorAxisLength",
        "shape3d.least_axis_length": "original_shape_LeastAxisLength",
        #################### First Order Statistics #######################
        "fos.energy": "original_firstorder_Energy",
        "fos.total_energy": "original_firstorder_TotalEnergy",
        "fos.entropy": "original_firstorder_Entropy",
        "fos.minimum": "original_firstorder_Minimum",
        "fos.10th_percentile": "original_firstorder_10Percentile",
        "fos.90th_percentile": "original_firstorder_90Percentile",
        "fos.maximum": "original_firstorder_Maximum",
        "fos.mean": "original_firstorder_Mean",
        "fos.median": "original_firstorder_Median",
        "fos.interquartile_range": "original_firstorder_InterquartileRange",
        "fos.range": "original_firstorder_Range",
        "fos.mean_absolute_deviation": "original_firstorder_MeanAbsoluteDeviation",
        "fos.robust_mean_absolute_deviation": "original_firstorder_RobustMeanAbsoluteDeviation",
        "fos.root_mean_squared": "original_firstorder_RootMeanSquared",
        "fos.skewness": "original_firstorder_Skewness",
        "fos.kurtosis": "original_firstorder_Kurtosis",
        "fos.variance": "original_firstorder_Variance",
        "fos.uniformity": "original_firstorder_Uniformity",
        ############## Gray Level Cooccurence Matrix (GLCM) #################
        "glcm.autocorrelation": "original_glcm_Autocorrelation",
        "glcm.joint_average": "original_glcm_JointAverage",
        "glcm.cluster_prominence": "original_glcm_ClusterProminence",
        "glcm.cluster_shade": "original_glcm_ClusterShade",
        "glcm.cluster_tendency": "original_glcm_ClusterTendency",
        "glcm.contrast": "original_glcm_Contrast",
        "glcm.correlation": "original_glcm_Correlation",
        "glcm.difference_average": "original_glcm_DifferenceAverage",
        "glcm.difference_entropy": "original_glcm_DifferenceEntropy",
        "glcm.difference_variance": "original_glcm_DifferenceVariance",
        "glcm.joint_energy": "original_glcm_JointEnergy",
        "glcm.joint_entropy": "original_glcm_JointEntropy",
        "glcm.imc1": "original_glcm_Imc1",
        "glcm.imc2": "original_glcm_Imc2",
        "glcm.idm": "original_glcm_Idm",
        "glcm.mcc": "original_glcm_MCC",
        "glcm.idmn": "original_glcm_Idmn",
        "glcm.id": "original_glcm_Id",
        "glcm.idn": "original_glcm_Idn",
        "glcm.inverse_variance": "original_glcm_InverseVariance",
        "glcm.max_probability": "original_glcm_MaximumProbability",
        "glcm.sum_entropy": "original_glcm_SumEntropy",
        "glcm.sum_squares": "original_glcm_SumSquares",
        ############## Gray Level Run Length Matrix (GLRLM) #################
        "glrlm.short_run_emphasis": "original_glrlm_ShortRunEmphasis",
        "glrlm.long_run_emphasis": "original_glrlm_LongRunEmphasis",
        "glrlm.gray_level_non_uniformity": "original_glrlm_GrayLevelNonUniformity",
        "glrlm.gray_level_non_uniformity_normalied": "original_glrlm_GrayLevelNonUniformityNormalized",
        "glrlm.run_length_non_uniformity": "original_glrlm_RunLengthNonUniformity",
        "glrlm.run_length_non_uniformity_normalized": "original_glrlm_RunLengthNonUniformityNormalized",
        "glrlm.run_percentage": "original_glrlm_RunPercentage",
        "glrlm.gray_level_variance": "original_glrlm_GrayLevelVariance",
        "glrlm.run_variance": "original_glrlm_RunVariance",
        "glrlm.run_entropy": "original_glrlm_RunEntropy",
        "glrlm.low_gray_level_run_empathsis": "original_glrlm_LowGrayLevelRunEmphasis",
        "glrlm.hight_gray_level_run_empathsis": "original_glrlm_HighGrayLevelRunEmphasis",
        "glrlm.short_run_low_gray_level_emphasis": "original_glrlm_ShortRunLowGrayLevelEmphasis",
        "glrlm.short_run_high_gray_level_emphasis": "original_glrlm_ShortRunHighGrayLevelEmphasis",
        "glrlm.long_run_low_gray_level_emphasis": "original_glrlm_LongRunLowGrayLevelEmphasis",
        "glrlm.long_run_high_gray_level_emphasis": "original_glrlm_LongRunHighGrayLevelEmphasis",
        ############## Gray Level Dependence Matrix (GLDM)  #################
        "gldm.small_dependence_emphasis": "original_gldm_SmallDependenceEmphasis",
        "gldm.large_dependence_emphasis": "original_gldm_LargeDependenceEmphasis",
        "gldm.gray_level_non_uniformity": "original_gldm_GrayLevelNonUniformity",
        "gldm.dependence_non_uniformity": "original_gldm_DependenceNonUniformity",
        "gldm.dependence_non_uniformity_normalized": "original_gldm_DependenceNonUniformityNormalized",
        "gldm.gray_level_variance": "original_gldm_GrayLevelVariance",
        "gldm.dependence_variance": "original_gldm_DependenceVariance",
        "gldm.dependence_entropy": "original_gldm_DependenceEntropy",
        "gldm.low_gray_level_emphasis": "original_gldm_LowGrayLevelEmphasis",
        "gldm.high_gray_level_emphasis": "original_gldm_HighGrayLevelEmphasis",
        "gldm.small_dependence_low_gray_level_emphasis": "original_gldm_SmallDependenceLowGrayLevelEmphasis",
        "gldm.small_dependence_high_gray_level_emphasis": "original_gldm_SmallDependenceHighGrayLevelEmphasis",
        "gldm.large_dependence_low_gray_level_emphasis": "original_gldm_LargeDependenceLowGrayLevelEmphasis",
        "gldm.large_dependence_high_gray_level_emphasis": "original_gldm_LargeDependenceHighGrayLevelEmphasis",
        ############## Gray Level Size Zone Matrix (glszm) #################
        "glszm.small_area_emphasis": "original_glszm_SmallAreaEmphasis",
        "glszm.large_area_emphasis": "original_glszm_LargeAreaEmphasis",
        "glszm.gray_level_non_uniformity": "original_glszm_GrayLevelNonUniformity",
        "glszm.gray_level_non_uniformity_normalized": "original_glszm_GrayLevelNonUniformityNormalized",
        "glszm.size_zone_non_uniformity": "original_glszm_SizeZoneNonUniformity",
        "glszm.size_zone_non_uniformity_normalized": "original_glszm_SizeZoneNonUniformityNormalized",
        "glszm.zone_percentage": "original_glszm_ZonePercentage",
        "glszm.gray_level_variance": "original_glszm_GrayLevelVariance",
        "glszm.zone_variance": "original_glszm_ZoneVariance",
        "glszm.zone_entropy": "original_glszm_ZoneEntropy",
        "glszm.low_gray_level_zone_emphasis": "original_glszm_LowGrayLevelZoneEmphasis",
        "glszm.high_gray_level_zone_emphasis": "original_glszm_HighGrayLevelZoneEmphasis",
        "glszm.small_area_low_gray_level_emphasis": "original_glszm_SmallAreaLowGrayLevelEmphasis",
        "glszm.small_area_high_gray_level_emphasis": "original_glszm_SmallAreaHighGrayLevelEmphasis",
        "glszm.large_area_low_gray_level_emphasis": "original_glszm_LargeAreaLowGrayLevelEmphasis",
        "glszm.large_area_high_gray_level_emphasis": "original_glszm_LargeAreaHighGrayLevelEmphasis",
        ############## Neighbouring Gray Tone Difference Matrix (ngtdm) #################
        "ngtdm.coarseness": "original_ngtdm_Coarseness",
        "ngtdm.contrast": "original_ngtdm_Contrast",
        "ngtdm.busyness": "original_ngtdm_Busyness",
        "ngtdm.complexity": "original_ngtdm_Complexity",
        "ngtdm.strength": "original_ngtdm_Strength",
    }

    return features


def test_get_radiomics_features():
    param_path = './Params.yaml'
    image_path = '../data/Decathlon/images/pancreas_005.nii.gz'
    tumor_path = '../data/Decathlon/tumor/pancreas_005.nii.gz'

    feature, keys = get_radiomics_features(image_path=image_path, label_path=tumor_path, param_path=param_path)
    print(len(feature))
    for key in keys:
        print("{:50s} - {}".format(key, feature[key]))


def test_translate_features():
    param_path = './Params.yaml'
    image_path = '../data/Decathlon/image/pancreas_005.nii.gz'
    tumor_path = '../data/Decathlon/tumor/pancreas_005.nii.gz'
    keys = [
        "shape3d.mesh_volume",
        "shape3d.voxel_volume",
        "shape3d.surface_area",
        "shape3d.surfaceVolumeRatio",
        "shape3d.sphericity",
        "shape3d.max_3d_diameter",
        "shape3d.max_2d_diameter",
        "shape3d.max_2d_diameter_column",
        "shape3d.max_2d_diameter_row",
        "shape3d.major_axis_length",
        "shape3d.minor_axis_length",
        "shape3d.least_axis_length",
        "fos.energy",
        "fos.total_energy",
        "fos.entropy",
        "fos.minimum",
        "fos.10th_percentile",
        "fos.90th_percentile",
        "fos.maximum",
        "fos.mean",
        "fos.median",
        "fos.interquartile_range",
        "fos.range",
        "fos.mean_absolute_deviation",
        "fos.robust_mean_absolute_deviation",
        "fos.root_mean_squared",
        "fos.skewness",
        "fos.kurtosis",
        "fos.variance",
        "fos.uniformity",
        "glcm.autocorrelation",
        "glcm.joint_average",
        "glcm.cluster_prominence",
        "glcm.cluster_shade",
        "glcm.cluster_tendency",
        "glcm.contrast",
        "glcm.correlation",
        "glcm.difference_average",
        "glcm.difference_entropy",
        "glcm.difference_variance",
        "glcm.joint_energy",
        "glcm.joint_entropy",
        "glcm.imc1",
        "glcm.imc2",
        "glcm.idm",
        "glcm.mcc",
        "glcm.idmn",
        "glcm.id",
        "glcm.idn",
        "glcm.inverse_variance",
        "glcm.max_probability",
        "glcm.sum_entropy",
        "glcm.sum_squares",
        "glrlm.short_run_emphasis",
        "glrlm.long_run_emphasis",
        "glrlm.gray_level_non_uniformity",
        "glrlm.gray_level_non_uniformity_normalied",
        "glrlm.run_length_non_uniformity",
        "glrlm.run_length_non_uniformity_normalized",
        "glrlm.run_percentage",
        "glrlm.gray_level_variance",
        "glrlm.run_variance",
        "glrlm.run_entropy",
        "glrlm.low_gray_level_run_empathsis",
        "glrlm.hight_gray_level_run_empathsis",
        "glrlm.short_run_low_gray_level_emphasis",
        "glrlm.short_run_high_gray_level_emphasis",
        "glrlm.long_run_low_gray_level_emphasis",
        "glrlm.long_run_high_gray_level_emphasis",
        "gldm.small_dependence_emphasis",
        "gldm.large_dependence_emphasis",
        "gldm.gray_level_non_uniformity",
        "gldm.dependence_non_uniformity",
        "gldm.dependence_non_uniformity_normalized",
        "gldm.gray_level_variance",
        "gldm.dependence_variance",
        "gldm.dependence_entropy",
        "gldm.low_gray_level_emphasis",
        "gldm.high_gray_level_emphasis",
        "gldm.small_dependence_low_gray_level_emphasis",
        "gldm.small_dependence_high_gray_level_emphasis",
        "gldm.large_dependence_low_gray_level_emphasis",
        "gldm.large_dependence_high_gray_level_emphasis",
        "glszm.small_area_emphasis",
        "glszm.large_area_emphasis",
        "glszm.gray_level_non_uniformity",
        "glszm.gray_level_non_uniformity_normalized",
        "glszm.size_zone_non_uniformity",
        "glszm.size_zone_non_uniformity_normalized",
        "glszm.zone_percentage",
        "glszm.gray_level_variance",
        "glszm.zone_variance",
        "glszm.zone_entropy",
        "glszm.low_gray_level_zone_emphasis",
        "glszm.high_gray_level_zone_emphasis",
        "glszm.small_area_low_gray_level_emphasis",
        "glszm.small_area_high_gray_level_emphasis",
        "glszm.large_area_low_gray_level_emphasis",
        "glszm.large_area_high_gray_level_emphasis",
        "ngtdm.coarseness",
        "ngtdm.contrast",
        "ngtdm.busyness",
        "ngtdm.complexity",
        "ngtdm.strength",
    ]
    translate = translate_keys()

    feature, _ = get_radiomics_features(image_path=image_path, label_path=tumor_path, param_path=param_path)

    for key in keys:
        print("{:50s} - {}".format(key, feature[translate[key]]))


if __name__ == '__main__':
    test_get_radiomics_features()
    # test_translate_features()
