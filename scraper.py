import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By

def scrape_imdb_movies(url, output_file="movies.csv"):
  
    driver = webdriver.Chrome()
    driver.get(url)
    time.sleep(3)

    movie_names = []
    storylines = []

    movies = driver.find_elements(By.CSS_SELECTOR, ".ipc-title-link-wrapper")
    plots = driver.find_elements(By.CSS_SELECTOR, ".ipc-html-content-inner-div")

    for movie, plot in zip(movies, plots):
        movie_names.append(movie.text.strip())
        storylines.append(plot.text.strip())

    driver.quit()

    df = pd.DataFrame({"Movie Name": movie_names, "Storyline": storylines})
    df.to_csv(output_file, index=False)
    print(f"✅ Data saved to {output_file}")

if __name__ == "__main__":
    scrape_imdb_movies("https://www.imdb.com/movies-coming-soon/2024/")
