use kd_tree::KdPoint;
use rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder,SentenceEmbeddingsModelType};
use serde::{Deserialize};
use std::io;
use std::fs;
use std::error;
use std::result;

type Result<T> = result::Result<T, Box<dyn error::Error>>;


#[derive(Debug, Deserialize)]
pub struct Library {
    pub books: Vec<Book>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Book {
    pub wikipedia_article_id: isize,
    pub freebase_id: String,
    pub title: String,
    pub author: String,
    pub publication_date: String,
    pub genres: String,
    pub summary: String,
}

impl Book {
    pub fn to_embedded(&self, embeddings: [f32; 384]) -> EmbeddedBook {
        EmbeddedBook {
            title: self.title.clone(),
            author: self.author.clone(),
            embeddings: embeddings,
        }
    }
}

#[derive(Debug, )]
pub struct EmbeddedBook {
    pub title: String,
    pub author: String,
    pub embeddings: [f32; 384],
}

impl EmbeddedBook {
    pub fn topic(embeddings: [f32; 384]) -> Self {
        Self {
            title: String::new(),
            author: String::new(),
            embeddings: embeddings,
        }
    }
}

impl KdPoint for EmbeddedBook {
    type Scalar = f32;
    type Dim = typenum::U2; // 2 dimensions

    fn at(&self, k: usize) -> f32 {
        self.embeddings[k]
    }
}

fn main() -> Result<()> {
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2).create_model()?;
    let f = fs::File::open("booksummaries.txt")?;
    let reader = io::BufReader::new(f);

    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_reader(reader);
    
    let mut embedded_books = Vec::new();
    for result in rdr.deserialize() {
        // Notice that we need to provide a type hint for automatic
        // deserialization.
        let book: Book = result?;
        let embeddings = model.encode(&[book.title.clone()])?;
        let embedding: [f32; 384] = embeddings[0].as_slice().try_into().expect("bad array size");
        embedded_books.push(book.to_embedded(embedding));
    }

    let kd_tree = kd_tree::KdSlice::sort_by(&mut embedded_books, |item1, item2, k| {
        item1.embeddings[k]
            .partial_cmp(&item2.embeddings[k])
            .unwrap()
    });

    println!("Type your query below...\n");
    let mut query = String::new();
    io::stdin().read_line(&mut query)?;
    println!("Querying: {}", query);
    let rich_embeddings = model.encode(&[query])?;
    let rich_embedding = rich_embeddings[0].as_slice().try_into().expect("bad array size");
    let rich_topic = EmbeddedBook::topic(rich_embedding);
    let nearests = kd_tree.nearests(&rich_topic, 10);
    for nearest in nearests {
        println!("nearest: {:?}", nearest.item.title);
        println!("distance: {:?}", nearest.squared_distance);
    }

    Ok(())
}
