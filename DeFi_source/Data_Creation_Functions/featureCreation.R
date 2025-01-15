library(dplyr)
library(readr)

select <- dplyr::select
rename <- dplyr::rename



# The purpose of this script is to take raw transaction data and generate features at each transaction that we think could be useful covariates for survival analysis. 

# Load the transaction data:
rawTransactions <- read_rds("/data/IDEA_DeFi_Research/Data/Lending_Protocols/Aave/V2/Mainnet/transactions.rds") 

# For ease of interpretation, we are going to rename the 'onBehalfOf' column to be 'user', since the 'onBehalfOf' column
# represents the address of the account that is actually being affected by the transaction.
rawTransactions <- rawTransactions %>%
  select(-user) %>%
  rename(user = onBehalfOf) %>%
  select(-userAlias, -onBehalfOfAlias)
  

# Separate into distinct tables by transaction type:
borrows <- rawTransactions %>%
  filter(type == "borrow") %>%
  mutate(priceInUSD = amountUSD / amount)

deposits <- rawTransactions %>%
  filter(type == "deposit")%>%
  mutate(priceInUSD = amountUSD / amount)

repays <- rawTransactions %>%
  filter(type == "repay")%>%
  mutate(priceInUSD = amountUSD / amount)

withdraws <- rawTransactions %>%
  filter(type == "withdraw")%>%
  mutate(priceInUSD = amountUSD / amount)

liquidations <- rawTransactions %>%
  filter(type == "liquidation")

# Initial processing of transactions to include "account liquidated" and "liquidation performed" as separate transactions:
liquidationsPerformed <- liquidations %>%
  select(-user) %>%
  rename(user = liquidator) %>%
  mutate(reserve = principalReserve,
         amount = principalAmount,
         amountUSD = principalAmountUSD,
         amountETH = principalAmountETH)%>%
  mutate(priceInUSD = amountUSD / amount)

accountsLiquidatedCollateral <- liquidations %>%
  mutate(reserve = collateralReserve,
         amount = collateralAmount,
         amountUSD = collateralAmountUSD,
         amountETH = collateralAmountETH)%>%
  mutate(priceInUSD = amountUSD / amount)

accountsLiquidatedPrincipal <- liquidations %>%
  mutate(reserve = principalReserve,
         amount = principalAmount,
         amountUSD = principalAmountUSD,
         amountETH = principalAmountETH) %>%
  mutate(priceInUSD = amountUSD / amount)


#####
# User-Reserve level features:
#####
# First, let's calculate the supply balance for user-reserve combinations. 
# The transaction types that affect supply balances are deposits, withdraws, and accountsLiquidated (the amountCollateral affects the user's supply):
deposits <- deposits %>%
  group_by(user, reserve) %>%
  mutate(depositedBalance = cumsum(amount))

withdrawsAndAccountsLiquidated <- bind_rows(withdraws, accountsLiquidatedCollateral) %>%
  arrange(timestamp) %>%
  group_by(user, reserve) %>%
  mutate(withdrawnBalance = cumsum(amount))

supplyBalances <- bind_rows(deposits, withdrawsAndAccountsLiquidated) %>%
  arrange(timestamp) %>%
  group_by(user, reserve) %>%
  fill(depositedBalance) %>%
  fill(withdrawnBalance)

supplyBalances[c('depositedBalance', 'withdrawnBalance')][is.na(supplyBalances[c('depositedBalance', 'withdrawnBalance')])] <- 0

supplyBalances <- supplyBalances %>%
  mutate(userReserveSupply = depositedBalance - withdrawnBalance) %>%
  mutate(userReserveSupplyUSD = userReserveSupply * priceInUSD)

#####
# Next, let's do the same thing for the borrows and repayments:
#####
# The transaction types that affect a user's loan balance are borrows, repays, and accountsLiqudated (the amountPrincipal pays off a portion of their loan).
# Note that these calculations do not account for interest on their loans, and don't perfectly handle whether a user pays off their own account or someone else's,
# so some running account balances can end up looking wacky. I don't know whether we have enough information from the Aave transactions to handle this properly.

borrows <- borrows %>%
  arrange(timestamp) %>%
  group_by(user, reserve) %>%
  mutate(loanBalance = cumsum(amount), numReserveBorrows = row_number()) # Calculate how much of this currency the user has borrowed in total at any point

repaysAndLiquidationsPerformed <- bind_rows(repays, accountsLiquidatedPrincipal) %>%
  arrange(timestamp) %>%
  group_by(user, reserve) %>%
  mutate(repaidBalance = cumsum(amount), numReserveRepays = row_number()) # Calculate how much of this currency the user has repaid in total at any point, including repayments from their account being liquidated

loanBalances <- bind_rows(borrows, repaysAndLiquidationsPerformed) %>%
  arrange(timestamp) %>%
  group_by(user, reserve) %>%
  fill(loanBalance) %>%
  fill(repaidBalance) # Combine these tables and perform a downward fill to keep the most up-to-date estimates of their account balances


loanBalances[c('loanBalance', 'repaidBalance')][is.na(loanBalances[c('loanBalance','repaidBalance')])] <- 0
  
loanBalances <- loanBalances %>%
  mutate(userReserveDebt = loanBalance - repaidBalance) %>%
  mutate(userReserveDebtUSD = userReserveDebt * priceInUSD)


#####