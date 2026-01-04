from tenacity import retry, stop_after_attempt


@retry(stop=stop_after_attempt(5))
def stop_after_5_attempts():
    print("I am stopping after 5 attempts")
    raise Exception

if __name__ == "__main__":
    stop_after_5_attempts()