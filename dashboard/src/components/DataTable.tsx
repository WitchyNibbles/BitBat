import type { ReactNode } from 'react';
import styles from './DataTable.module.css';

interface Column {
  key: string;
  label: string;
  align?: 'left' | 'right' | 'center';
}

interface DataTableProps {
  columns: Column[];
  rows: Record<string, ReactNode>[];
}

export function DataTable({ columns, rows }: DataTableProps) {
  return (
    <div className={styles.wrapper}>
      <table className={styles.table}>
        <thead>
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                className={`${styles.th} ${styles[col.align ?? 'left']}`}
              >
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className={styles.row}>
              {columns.map((col) => (
                <td
                  key={col.key}
                  className={`${styles.td} ${styles[col.align ?? 'left']}`}
                >
                  {row[col.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
