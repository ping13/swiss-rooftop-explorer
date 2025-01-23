import * as duckdb from '@duckdb/duckdb-wasm';
import duckdb_wasm from '@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url';
import mvp_worker from '@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url';

// Initialize DuckDB configuration
const MANUAL_BUNDLES = {
    mvp: {
        mainModule: duckdb_wasm,
        mainWorker: mvp_worker
    }
};

export class AddressDatabase {
    constructor() {
        this.db = null;
        this.conn = null;
        this.worker = null;
    }

    async initialize() {
        try {
            // Select bundle and create worker
            const bundle = await duckdb.selectBundle(MANUAL_BUNDLES);
            this.worker = new Worker(bundle.mainWorker);
            
            const logger = new duckdb.ConsoleLogger();
            this.db = new duckdb.AsyncDuckDB(logger, this.worker);
            
            await this.db.instantiate(bundle.mainModule);
            
            // Create connection
            this.conn = await this.db.connect();
            
            console.log('DuckDB initialized successfully');
        } catch (error) {
            console.error('Failed to initialize DuckDB:', error);
            throw error;
        }
    }

    async getAddressesByZipCode(zipCode, columns = ['*'], pq_file = "addresses_full.parquet") {
        try {
            const columnSelection = Array.isArray(columns) ? columns.join(', ') : '*';
            const query = `
                SELECT ${columnSelection} 
                FROM parquet_scan('https://ping13.net/data/${pq_file}') 
                WHERE dplz4 = ${zipCode}
            `;
            
            const result = await this.conn.query(query);
            
            // Convert Arrow Table to JSON
            const jsonArray = [];
            for (let rowIdx = 0; rowIdx < result.numRows; rowIdx++) {
                const row = {};
                for (const field of result.schema.fields) {
                    row[field.name] = result.getChildAt(result.schema.fields.indexOf(field)).get(rowIdx);
                }
                jsonArray.push(row);
            }
            
            return jsonArray;
        } catch (error) {
            console.error('Error querying addresses:', error);
            throw error;
        }
    }

    async getRandomZipCode(pq_file = "addresses.parquet") {
        try {
            const query = `
                SELECT DISTINCT dplz4 
                FROM parquet_scan('https://ping13.net/data/${pq_file}')
                ORDER BY random()
                LIMIT 1
            `;
            
            const result = await this.conn.query(query);
            return result.getChildAt(0).get(0);  // Get the first value from the first column
        } catch (error) {
            console.error('Error getting random ZIP code:', error);
            throw error;
        }
    }

    async close() {
        if (this.conn) await this.conn.close();
        if (this.db) await this.db.terminate();
    }
}

export async function searchAddresses(zipCode, addressDB) {
    try {
        const results = await addressDB.getAddressesByZipCode(zipCode,
            ['STRNAME', 'DEINR', 'DPLZNAME', 'latitude', 'longitude'],
            "addresses.parquet"
        );
        console.log(results);
        return results;
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}
