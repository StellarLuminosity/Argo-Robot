class Portfolio:
    """
    A class to represent a financial portfolio consisting of assets and fiat currency.

    Attributes:
    asset (float): The amount of assets in the portfolio.
    fiat (float): The amount of fiat currency in the portfolio.
    interest_asset (float): The interest on the assets.
    interest_fiat (float): The interest on the fiat currency.
    """

    def __init__(self, asset: float, fiat: float, interest_asset: float = 0, interest_fiat: float = 0):
        """
        Constructs all the necessary attributes for the portfolio object.

        Parameters:
        asset (float): The amount of assets in the portfolio.
        fiat (float): The amount of fiat currency in the portfolio.
        interest_asset (float): The interest on the assets.
        interest_fiat (float): The interest on the fiat currency.
        """
        self.asset: float = asset
        self.fiat: float = fiat
        self.interest_asset: float = interest_asset
        self.interest_fiat: float = interest_fiat

    def valorisation(self, price: float) -> float:
        """
        Calculates the total value of the portfolio.

        Parameters:
        price (float): The current price of the asset.

        Returns:
        float: The total value of the portfolio.
        """
        return sum([
            self.asset * price,
            self.fiat,
            -self.interest_asset * price,
            -self.interest_fiat
        ])

    def real_position(self, price: float) -> float:
        """
        Calculates the real position of the portfolio, considering the interest.

        Parameters:
        price (float): The current price of the asset.

        Returns:
        float: The real position of the portfolio.
        """
        return (self.asset - self.interest_asset) * price / self.valorisation(price)

    def position(self, price: float) -> float:
        """
        Calculates the position of the portfolio without considering the interest.

        Parameters:
        price (float): The current price of the asset.

        Returns:
        float: The position of the portfolio.
        """
        return self.asset * price / self.valorisation(price)

    def trade_to_position(self, position: float, price: float, trading_fees: float) -> None:
        """
        Adjusts the portfolio to a new position by trading assets and fiat currency.

        This function first repays any interest on the current position, then proceeds
        to trade assets and fiat currency to achieve the desired position.

        Parameters:
        position (float): The target position ratio of assets to total portfolio value.
                          A value between 0 and 1 represents a long position, while a value
                          greater than 1 represents a leveraged long position, and a value
                          less than 0 represents a short position.
        price (float): The current price of the asset.
        trading_fees (float): The trading fees as a fraction of the trade value.

        Returns:
        None
        """
        # Repay interest
        current_position = self.position(price)
        interest_reduction_ratio = 1.0
        if position <= 0 and current_position < 0:
            interest_reduction_ratio = min(1, position / current_position)
        elif position >= 1 and current_position > 1:
            interest_reduction_ratio = min(1, (position - 1) / (current_position - 1))
        if interest_reduction_ratio < 1:
            self.asset = self.asset - (1 - interest_reduction_ratio) * self.interest_asset
            self.fiat = self.fiat - (1 - interest_reduction_ratio) * self.interest_fiat
            self.interest_asset = interest_reduction_ratio * self.interest_asset
            self.interest_fiat = interest_reduction_ratio * self.interest_fiat

        # Proceed to trade
        asset_trade = (position * self.valorisation(price) / price - self.asset)
        if asset_trade > 0:
            asset_trade = asset_trade / (1 - trading_fees + trading_fees * position)
            asset_fiat = -asset_trade * price
            self.asset = self.asset + asset_trade * (1 - trading_fees)
            self.fiat = self.fiat + asset_fiat
        else:
            asset_trade = asset_trade / (1 - trading_fees * position)
            asset_fiat = -asset_trade * price
            self.asset = self.asset + asset_trade
            self.fiat = self.fiat + asset_fiat * (1 - trading_fees)

    def update_interest(self, borrow_interest_rate: float) -> None:
        """
        Updates the interest on the assets and fiat currency based on the borrow interest rate.

        Parameters:
        borrow_interest_rate (float): The interest rate for borrowing.

        Returns:
        None
        """
        self.interest_asset = max(0, -self.asset) * borrow_interest_rate
        self.interest_fiat = max(0, -self.fiat) * borrow_interest_rate

    def __str__(self) -> str:
        """
        Returns a string representation of the portfolio.

        Returns:
        str: A string representation of the portfolio.
        """
        return f"{self.__class__.__name__}({self.__dict__})"

    def describe(self, price: float) -> None:
        """
        Prints the value and position of the portfolio.

        Parameters:
        price (float): The current price of the asset.

        Returns:
        None
        """
        print("Value : ", self.valorisation(price), "Position : ", self.position(price))

    def get_portfolio_distribution(self) -> None:
        """
        Placeholder method for getting the portfolio distribution.

        Returns:
        None
        """
        pass

class TargetPortfolio(Portfolio):
    """
    A class to represent a target financial portfolio, inheriting from Portfolio.

    Attributes:
    position (float): The target position ratio of assets to total portfolio value.
    value (float): The total value of the portfolio.
    price (float): The current price of the asset.
    """

    def __init__(self, position, value, price):
        """
        Constructs all the necessary attributes for the target portfolio object.

        Parameters:
        position (float): The target position ratio of assets to total portfolio value.
        value (float): The total value of the portfolio.
        price (float): The current price of the asset.
        """
        super().__init__(
            asset = position * value / price,
            fiat = (1-position) * value,
            interest_asset = 0,
            interest_fiat = 0
        )