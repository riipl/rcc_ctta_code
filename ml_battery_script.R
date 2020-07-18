# GENERALIZED MACHINE LEARNING BATTERY
# Akshay Jaggi, Sandy Napel
# Department of Radiology, Stanford University School of Medicine

# This is a framework for supervised machine learning tasks
# The overhead function is at the bottom!
# Start there and then move up through the helper functions as needed


# Majority Vote
# Input:    Importances
# Output:   Results of the majority vote
# Does:     For each combination of parameters
#           Combine all the trials into a data frame
majority_vote = function(importances) {
  results = list()
  for (n in names(importances)) {
    results[[n]] = list()
    for (selection in names(importances[[n]])) {
      results[[n]][[selection]] = list()
      for (model in names(importances[[n]][[selection]])) {
        voting = data.frame()
        counter = 0
        for (trial in names(importances[[n]][[selection]][[model]])) {
          bottom = importances[[n]][[selection]][[model]][[trial]]
          rotated_better = t(t(bottom$importance)[1, ])
          names(rotated_better) = colnames(rotated_better)
          row.names(rotated_better) = trial
          if (counter == 0) {
            voting = rotated_better
          } else {
            voting1 <<- voting
            voting2 <<- rotated_better
            voting = bind_rows(voting, rotated_better)
          }
          counter = counter + 1
        }
        results[[n]][[selection]][[model]] = voting
      }
    }
  }
  return(results)
}

# Reshape Nests
# Input:    Importances
# Output:   Reshaped importances
# Does:     Super hacky helper function to un-nest and re-nest the importances
#           Necessary such that all the trials are grouped together
reshape_nests = function(importances) {
  reshaped_nests = list()
  counter = 0
  for (trial in names(importances)) {
    for (n in names(importances[[trial]])) {
      if (counter == 0) {
        reshaped_nests[[n]] = list()
      }
      for (selection in names(importances[[trial]][[n]])) {
        if (counter == 0) {
          reshaped_nests[[n]][[selection]] = list()
        }
        out = importances[[trial]][[n]][[selection]]
        for (name in names(out)) {
          if (counter == 0) {
            reshaped_nests[[n]][[selection]][[name]] = list()
          }
          reshaped_nests[[n]][[selection]][[name]][[trial]] = out[[name]]
        }
        
      }
    }
    counter = counter + 1
  }
  return(reshaped_nests)
}

# Majority Vote Overhead
# Input:    Importances
# Output:   Results of importance voting
# Does:     First reshapes the nesting of  the importances to group all trials
#           Then runs majority voting across all the trials for each combination of parameters
majority_vote_overhead = function(importances) {
  importance_reshaped = reshape_nests(importances)
  results = majority_vote(importance_reshaped)
  return(results)
}

# Get ROCs
# Input:    List of Models
#           Testing Data
# Output:   Summaries of model performance for each model type
# Does:     Get predictions and probabilities for all testing data using all models
#           Then for each model, produce the roc, sensitivity, and specificity. Then return
get_rocs = function(model_list, x_test, y_test, model_names) {
  probs = predict(model_list, newdata = x_test, type = "prob")
  preds = predict(model_list, newdata = x_test)
  test_summaries = list()
  for (model in model_names) {
    this_models_probs = probs[[model]]
    this_models_preds = preds[[model]]
    if(model == "ensemble") {
      test_set = data.frame("obs" = y_test, 
                            "X0" = this_models_probs, 
                            "X1" = 1- this_models_probs,
                            "pred" = this_models_preds)
    } else {
      test_set = data.frame("obs" = y_test, this_models_probs, "pred" = this_models_preds)
    }
    
    test_summaries[[model]] = twoClassSummary(test_set, lev = levels(test_set$obs))
  }
  return(test_summaries)
}


# Modeling
# Input:    Training Data
#           Testing Data
#           Target Variable
#           Parameters
#           Selected Indices
# Output:   Summaries of the test performance
#           Importances of the variables in each classifier
# Does:     Run cross validated hyperparameter optimization for each modeling type
#           Train on all training data and report the importances of the variables
#           Test on the testing data and report the performances
modeling = function(training,
                    testing,
                    target,
                    parameters,
                    selected_indices) {
  model_list = list()
  importance_list = list()
  
  training_control = trainControl(
    method = "repeatedcv",
    number = 10,
    repeats = 1,
    savePredictions = "final",
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  )
  training_reduced = select(training, selected_indices)
  testing_reduced = select(testing, selected_indices)
  for (classifier in parameters$classifiers) {
    if(classifier == "ensemble") {
      to_ensemble = caretList(
        x = training_reduced,
        y =  training[[target]],
        trControl=training_control,
        metric = "ROC",
        methodList=c("xgbLinear", "ada")
      )
      model_list[[classifier]] = caretEnsemble(
        to_ensemble, 
        metric="ROC",
        trControl=trainControl(
          number=2,
          savePredictions = "final",
          classProbs = TRUE,
          summaryFunction=twoClassSummary,
        ))
    } else {
      model_list[[classifier]] = train(
        x = training_reduced,
        y = training[[target]],
        method = classifier,
        trControl = training_control,
        metric = "ROC"
      )
    }
    importance_list[[classifier]] = varImp(model_list[[classifier]])
  }
  test_summaries = get_rocs(model_list, testing_reduced, testing[[target]], parameters$classifiers)
  return(list("summaries" = test_summaries, "importances" = importance_list))
}


# Feature Selection
# Input:    Training Data
#           Target variable
#           Number of selected variables
#           Selection Method
# Output:   Return selected features
# Does:     Run Feature selection on the training data
feature_selection = function(training, target, n, selection_method) {
  target_vect = training[[target]]
  # Addresses requirment from wilcox split that the target be zero one
  target_vect_zero_one = as.numeric(target_vect) - 1
  switch(
    selection_method,
    "none"   =
      {
        return(1:nrow(training))
      },
    "wilcox" =
      {
        out = wilcox.selection.split(
          x = select(training, -target),
          y = target_vect_zero_one,
          split = generate.cv(length(target_vect_zero_one), 2)
        )
        return(out$ordering.split[1, 1:n])
      },
    "mRMR" =
      {
        out = MRMR(select(training, -target), target_vect, k = n)
        return(out$selection)
      }
  )
}


# ML Battery
# Input:  Training Data, Testing Data, Target variable name, parameters
# Output: Summary statistics for all combinations of parameters
#         List of importances for every selected feature
# Does:   Creates an array for statsfrom all combinations of the parameters
#         The array is p+1 dimensions where p is the number of dimensions (+1 since three stats)
#         Runs nested for loops for each parameter
#         Run Feature Selection
#         Using selected features, run modeling
ml_battery = function(training, testing, target, parameters) {
  summaries = array(dim = c(lengths(parameters), 3))
  importances = list()
  for (n in as.list(enumerate(parameters$features))) {
    print(n$value)
    sublist = list()
    for (selection in as.list(enumerate(parameters$selections))) {
      selected_features = feature_selection(training, target, n$value, selection$value)
      selected_indices = unlist(selected_features)
      model_out = modeling(training, testing, target, parameters, selected_indices)
      for (model in as.list(enumerate(names(model_out$summaries)))) {
        summaries[n$index, selection$index, model$index, ] = model_out$summaries[[model$value]]
      }
      sublist[[selection$value]] = model_out$importances
    }
    importances[[as.character(n$value)]] = sublist
  }
  return(list("summaries" = summaries, "importances" = importances))
}


# ML Battery Overhead
# Input:  Main Data Frame,
#         Name of Target Variable,
#         Number of Resampled Validations,
#         Parameter list,
#         Random Seed
# Output: List of all performance for each train test split
#         All Variabble importances
# Does:   Creates summary data frame for performance stats (ROC, sens, spec)
#         Data frame  stores performances for all combinations of parameters for all validations
#         Then splits the data and runs the ml battery and stores the performance summary
#         Performs majority voting on all importances
#         Returns summary statistics for all performance stats
ml_battery_overhead = function(all_data,target,number_of_validations,parameters,seed) {
  set.seed(seed)
  summaries = array(dim = c(
    "outer" = number_of_validations,
    lengths(parameters),
    "stats" = 3
  ))
  train_indices = createDataPartition(all_data[[target]], p = 0.7, times = number_of_validations)
  all_importances = list()
  for (i in 1:number_of_validations) {
    training = all_data[unlist(train_indices[i]), ]
    testing  = all_data[-unlist(train_indices[i]), ]
    battery_out = ml_battery(training, testing, target, parameters)
    summaries[i, , , , ] = battery_out$summaries
    all_importances[[paste("trial", i, sep = "_")]] = battery_out$importances
  }
  return(
    list(
      "means" =  summaries,
      "importances" = all_importances
    )
  )
}


ml_battery_overhead_icc = function(data_set1, data_set2, target, n_validations, parameters, seed) {
  set.seed(seed)
  summaries1 = array(dim = c(
    "outer" = n_validations,
    lengths(parameters),
    "stats" = 3
  ))
  summaries2 = array(dim = c(
    "outer" = n_validations,
    lengths(parameters),
    "stats" = 3
  ))
  all_importances1 = list()
  all_importances2 = list()
  train_indices = createDataPartition(data_set1[[target]], p = 0.7, times = n_validations)
  parameters_no_icc = parameters
  parameters_no_icc[["cutoffs"]] <- NULL
  for (i in 1:n_validations) {
    training1 = data_set1[unlist(train_indices[i]), ]
    testing1  = data_set1[-unlist(train_indices[i]), ]
    training2 = data_set2[unlist(train_indices[i]), ]
    testing2  = data_set2[-unlist(train_indices[i]), ]
    summaries_by_icc1 =  array(dim = c(lengths(parameters), 3))
    summaries_by_icc2 =  array(dim = c(lengths(parameters), 3))
    importances_by_icc1 = list()
    importances_by_icc2 = list()
    for(cutoff in as.list(enumerate(parameters$cutoffs))) {
      to_keep = icc_filter(training1, training2,cutoff$value)
      
      training_filt1 = drop_by_icc(training1, to_keep, target)
      testing_filt1 = drop_by_icc(testing1, to_keep, target)
      training_filt2 = drop_by_icc(training2, to_keep, target)
      testing_filt2 = drop_by_icc(testing2, to_keep, target)
        
      battery_out1 = ml_battery(training_filt1, testing_filt1, target, parameters_no_icc)
      battery_out2 = ml_battery(training_filt2, testing_filt2, target, parameters_no_icc)
      
      summaries_by_icc1[cutoff$index, , , , ] = battery_out1$summaries
      summaries_by_icc2[cutoff$index, , , , ] = battery_out2$summaries
      importances_by_icc1[[as.character(cutoff$value)]] = battery_out1$importances
      importances_by_icc2[[as.character(cutoff$value)]] = battery_out2$importances
    }
    summaries1[i, , , , , ] = summaries_by_icc1
    summaries2[i, , , , , ] = summaries_by_icc2
    all_importances1[[paste("trial", i, sep = "_")]] = importances_by_icc1
    all_importances2[[paste("trial", i, sep = "_")]] = importances_by_icc2
  }
  return(
    list(
      "means1" =  summaries1,
      "means2" =  summaries2,
      "importances1" = all_importances1,
      "importances2" = all_importances2
    )
  )
}

icc_filter = function(frame1, frame2, cutoff) {
  iccs = find_iccs(frame1, frame2)
  to_keep = icc_cut(iccs, cutoff)
}

find_iccs = function(data_set1, data_set2) {
  iccs = matrix(ncol=ncol(data_set1))
  for(i in 1:ncol(data_set1)) {
    tmp = irr::icc(t(rbind(data_set1[,i],data_set2[,i])))
    iccs[i] = tmp$value
  }
  iccs[is.na(iccs)] = 0
  iccs = data.frame(iccs)
  colnames(iccs) = colnames(data_set1)
  return(iccs)
}

icc_cut = function(iccs, cut_off) {
  to_keep = numeric()
  for(i in 1:length(iccs)) {
    if(abs(iccs[i]) >= cut_off) {
      to_keep = c(to_keep,i)
    }
  }
  return(to_keep)
}

drop_by_icc = function(data, to_keep, target) {
  data_filt = data[,to_keep]
  data_filt[[target]] = data[[target]]
  return(data_filt)
}