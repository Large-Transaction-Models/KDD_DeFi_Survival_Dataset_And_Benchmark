#####
# Create survival data for borrow-to-account-liquidation and borrow-to-liquidation-performed
#####

# To be very clear, this code is for computing survival data for the time it takes a user to get liquidated following a borrow:

liquidations <- transactions %>%
  filter(type == "liquidation") %>%
  mutate(principalAmountUSDQuartile = ntile(principalAmountUSD, 4),
         collateralAmountUSDQuartile = ntile(collateralAmountUSD, 4),
         principalCollateralCombination = paste0(principalReserve, ":", collateralReserve),
         principalType = case_when(principalReserve %in% stableCoins$symbol ~ "Stable",
                                   TRUE ~ "Non-Stable"),
         collateralType = case_when(collateralReserve %in% stableCoins$symbol ~ "Stable",
                                     TRUE ~ "Non-Stable"),
         principalCollateralTypeCombination = paste0(principalType, ":", collateralType))

indexCovariates <- c("id","reserve", "quarter", "coinType", "amount", "amountUSD", "amountUSDQuartile", "amountNativeQuartile")
outcomeCovariates <- c("principalAmountUSDQuartile", 
                       "collateralAmountUSDQuartile", 
                       "principalAmountUSD",
                       "principalAmount",
                       "collateralAmountUSD",
                       "collateralAmount",
                       "principalCollateralCombination", 
                       "principalType", 
                       "collateralType", 
                       "principalCollateralTypeCombination", 
                       "principalReserve", 
                       "collateralReserve")

timeStart = min(transactions$timestamp)
timeFinal = max(transactions$timestamp)


for(indexEvent in basicEventTypes$type){
  # Each index event should have its own directory. We create the appropriate directory
  # here, in case it doesn't already exist:
  dir.create(paste0(dataPath, str_to_title(indexEvent), "/Account Liquidated/"))
  
  survData <- NULL
  
  # Collect the index events and select the relevant features:
  indexEvents <- basicTransactions %>%
    filter(type == indexEvent,
           between(timestamp, timeStart, timeFinal)) %>%
    select(where(not_all_na)) %>%
    mutate(indexID = row_number(),
           indexTime = timestamp) %>%
    select(indexID, timestamp, indexTime, any_of(subjects), any_of(indexCovariates)) %>%
    data.table()
  
  # Collect the outcome events and select the relevant features:
  outcomeEvents <- liquidations %>%
    filter(between(timestamp, timeStart, timeFinal)) %>%
    select(where(not_all_na)) %>%
    mutate(outcomeID = row_number(),
           outcomeTime = timestamp) %>%
    select(outcomeID, timestamp, outcomeTime, user, liquidator, any_of(outcomeCovariates)) %>%
    data.table()
  
  uncensoredEvents <- outcomeEvents[indexEvents, on = c("user", "timestamp"), roll = -Inf] %>%
    filter(!is.na(outcomeID)) %>%
    mutate(timeDiff = outcomeTime - indexTime,
           status = 1) %>%
    select(indexID,
           indexTime,
           outcomeID,
           outcomeTime,
           timeDiff,
           status,
           user,
           any_of(indexCovariates),
           any_of(outcomeCovariates))
  
  # For right-censored events, we summarise using features from the index event and calculate the 
  # time-to-censorship using the timeFinal from above and the index time:
  rightCensoredEvents <- anti_join(indexEvents, uncensoredEvents, by = c("indexID")) %>%
    mutate(timeDiff = (timeFinal - indexTime)/1e3,
           status = 0) %>%
    select(indexID,
           indexTime,
           timeDiff,
           status,
           user,
           reserve,
           any_of(indexCovariates)) %>%
    distinct()
  
  leftCensoredEvents <- anti_join(outcomeEvents, uncensoredEvents, by = c("outcomeID")) %>%
    mutate(timeDiff = (outcomeTime - timeStart)/1e3,
           status = 0) %>%
    select(timeDiff,
           outcomeID,
           outcomeTime,
           status,
           liquidator,
           user,
           any_of(outcomeCovariates))
  
  survData <- bind_rows(survivalData, rightCensoredEvents, uncensoredEvents, leftCensoredEvents)%>%
    mutate(`Index Event` = toString(indexEvent),
           `Outcome Event` = "account liquidated")
  
  saveRDS(survData, paste0(dataPath, str_to_title(indexEvent), "/Account Liquidated/","survivalData_101024.rds"))
  
  #####
  # Create Index-to-Liquidation-Performed data:
  #####
  
  
  survData <- NULL
  
  dir.create(paste0(dataPath, str_to_title(indexEvent), "/Liquidation Performed/"))
  
  uncensoredEvents <- outcomeEvents[indexEvents, on = c("liquidator" = "user", "timestamp"), roll = -Inf] %>%
    filter(!is.na(outcomeID)) %>%
    mutate(timeDiff = outcomeTime - indexTime,
           status = 1) %>%
    select(indexID,
           indexTime,
           outcomeID,
           outcomeTime,
           timeDiff,
           status,
           user = liquidator,
           any_of(indexCovariates),
           any_of(outcomeCovariates))
  
  # For right-censored events, we summarise using features from the index event and calculate the 
  # time-to-censorship using the timeFinal from above and the index time:
  rightCensoredEvents <- anti_join(indexEvents, uncensoredEvents, by = c("indexID")) %>%
    mutate(timeDiff = timeFinal - indexTime,
           status = 0) %>%
    select(indexID,
           indexTime,
           timeDiff,
           status,
           user,
           reserve,
           any_of(indexCovariates)) %>%
    distinct()
  
  leftCensoredEvents <- anti_join(outcomeEvents, uncensoredEvents, by = c("outcomeID")) %>%
    mutate(timeDiff = outcomeTime - timeStart,
           status = 0) %>%
    select(timeDiff,
           outcomeID,
           outcomeTime,
           status,
           liquidatee = user,
           user = liquidator,
           any_of(outcomeCovariates))
  
  survData <- bind_rows(survData, rightCensoredEvents, uncensoredEvents, leftCensoredEvents) %>%
    mutate(`Index Event` = toString(indexEvent),
           `Outcome Event` = "liquidation performed")
    
  saveRDS(survData, paste0(dataPath, str_to_title(indexEvent), "/Liquidation Performed/","survivalData_101024.rds"))
  
  
}


