from game import Driver


if __name__ == "__main__":
    attempt = 0
    while True:
        attempt += 1
        driver = Driver(attempt)
        while driver.running:
            driver.main_loop()